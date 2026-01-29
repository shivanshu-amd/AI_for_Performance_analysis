#!/usr/bin/env python3
"""
Multi-Compiler Verification System
Reads compiler configurations from YAML file and verifies vectorization
across all configured compilers
"""

import sys
import os
import yaml
import subprocess
import csv
import tempfile
import re
from typing import Dict, List, Tuple
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from loop_analyzer import LoopAnalyzer, LoopInfo
from llm_vectorization_checker import LLMVectorizationChecker, create_loop_id


class CompilerConfig:
    """Represents a compiler configuration"""
    
    def __init__(self, config_dict):
        self.name = config_dict['name']
        self.module_path = config_dict.get('module_path', '')
        self.module_name = config_dict.get('module_name', '')
        self.compiler_command = config_dict['compiler_command']
        self.flags = config_dict['flags']
    
    def __repr__(self):
        return f"CompilerConfig({self.name}, {self.compiler_command})"


class MultiCompilerVerifier:
    """
    Verifies vectorization across multiple compilers
    Reads configuration from YAML file
    """
    
    def __init__(self, config_file='config/compiler_config.yaml'):
        """
        Initialize multi-compiler verifier
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        self.compilers = []
        self.llm_config = {}
        
        # Load configuration
        self._load_config()
        
        # Initialize LLM checker (on-prem only)
        self.llm_checker = LLMVectorizationChecker(
            model=self.llm_config.get('model', 'default'),
            api_key=self.llm_config.get('api_key')
        )
        
        self.loop_analyzer = LoopAnalyzer()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Set module path if specified
        module_path = config.get('module_path', '')
        if module_path:
            os.environ['MODULEPATH'] = module_path
            print(f"Set MODULEPATH to: {module_path}")
        
        # Load compiler configurations
        for compiler_dict in config.get('compilers', []):
            self.compilers.append(CompilerConfig(compiler_dict))
        
        # Load LLM configuration
        self.llm_config = config.get('llm', {})
        
        print(f"Loaded configuration from {self.config_file}")
        compiler_names = ', '.join([c.name for c in self.compilers])
        print(f"  Compilers: {len(self.compilers)} ({compiler_names})")
        print(f"  LLM Model: {self.llm_config.get('model', 'N/A')}")
        print()
    
    def _check_module_system(self) -> bool:
        """Check if module system is available"""
        try:
            # Try to find modulecmd
            result = subprocess.run(
                ['bash', '-c', 'type modulecmd'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _load_module(self, module_name: str, module_path: str = '') -> Tuple[bool, str]:
        """
        Load a module by executing modulecmd and applying environment changes
        
        Args:
            module_name: Module name to load
            module_path: Optional module path to set
            
        Returns:
            success: True if module loaded successfully
            message: Error message if failed
        """
        if not module_name:
            return True, "No module specified"
        
        try:
            # Build command to get environment changes from modulecmd
            # modulecmd python load <module> outputs Python code to modify environment
            cmd = f'modulecmd python load {module_name}'
            
            # Set MODULEPATH if provided
            env = os.environ.copy()
            if module_path:
                env['MODULEPATH'] = module_path
            
            result = subprocess.run(
                ['bash', '-c', cmd],
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode != 0:
                return False, f"Module not available or failed to load: {module_name}\n{result.stderr}"
            
            # Execute the Python code returned by modulecmd to update environment
            # This code typically contains os.environ updates
            try:
                exec(result.stdout)
                return True, f"Module loaded: {module_name}"
            except Exception as e:
                return False, f"Failed to apply module environment: {str(e)}"
                
        except Exception as e:
            return False, f"Error loading module: {str(e)}"
    
    def _check_compiler(self, compiler_config: CompilerConfig) -> Tuple[bool, str]:
        """
        Check if compiler is available
        
        Args:
            compiler_config: Compiler configuration
            
        Returns:
            available: True if compiler is available
            message: Status message
        """
        # Try to load module if module system is available
        if compiler_config.module_name and self._check_module_system():
            success, message = self._load_module(
                compiler_config.module_name,
                compiler_config.module_path
            )
            if not success:
                print(f"    Warning: {message}")
                # Continue anyway - compiler might be in PATH
            else:
                print(f"    {message}")
        
        # Check if compiler command exists (works with or without modules)
        try:
            result = subprocess.run(
                [compiler_config.compiler_command, '--version'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                return True, f"Compiler available: {version}"
            else:
                return False, f"Compiler not available: {compiler_config.compiler_command}"
                
        except FileNotFoundError:
            return False, f"Compiler not found: {compiler_config.compiler_command}"
    
    def _compile_and_get_report(self, source_file: str, compiler_config: CompilerConfig) -> Tuple[bool, str]:
        """
        Compile file and get vectorization report
        
        Args:
            source_file: Source file path
            compiler_config: Compiler configuration
            
        Returns:
            success: True if compilation succeeded
            report: Vectorization report
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp:
                tmp_obj = tmp.name
            
            # For Intel compiler, specify opt report file location explicitly
            optrpt_file = None
            if compiler_config.name == 'intel':
                with tempfile.NamedTemporaryFile(suffix='.optrpt', delete=False) as tmp_rpt:
                    optrpt_file = tmp_rpt.name
            
            # Build compile command
            cmd = [compiler_config.compiler_command] + compiler_config.flags.split()
            
            # Add opt report file flag for Intel
            if optrpt_file:
                cmd.append(f'-qopt-report-file={optrpt_file}')
            
            cmd.extend(['-c', source_file, '-o', tmp_obj])
            
            # Compile
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Get report (usually in stderr for GCC/Clang)
            report = result.stderr + result.stdout
            
            # Clean up object file
            if os.path.exists(tmp_obj):
                os.remove(tmp_obj)
            
            # Read Intel opt report file if specified
            if optrpt_file and os.path.exists(optrpt_file):
                with open(optrpt_file, 'r') as f:
                    optrpt_content = f.read()
                    report += '\n' + optrpt_content
                os.remove(optrpt_file)
            
            return True, report
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _parse_vectorization_status(self, report: str, source_file: str, line_number: int, compiler_name: str) -> bool:
        """
        Parse compiler report to check if loop was vectorized
        
        Args:
            report: Compiler report
            source_file: Source filename
            line_number: Loop line number
            compiler_name: Compiler name
            
        Returns:
            True if loop was vectorized
        """
        filename = os.path.basename(source_file)
        
        # Intel compiler opt report format (separate lines)
        # Intel creates multiversioned loops, so we need to check ALL versions
        # Pattern: "LOOP BEGIN at <path> (line_number, col)"
        intel_loop_pattern = rf'LOOP BEGIN at .*{re.escape(filename)}\s*\({line_number},\s*\d+\)'
        
        if re.search(intel_loop_pattern, report):
            # Found the loop - scan through ALL loop sections for this line
            # (Intel creates multiple versions: multiversioned v1, v2, remainder, etc.)
            lines = report.split('\n')
            in_target_loop = False
            current_depth = 0  # Track nesting depth to handle nested loops
            
            for line in lines:
                # Check if this is the start of ANY loop section
                if line.strip().startswith('LOOP BEGIN'):
                    # Check if it's our target line
                    if re.search(intel_loop_pattern, line):
                        in_target_loop = True
                        current_depth = 0
                    elif in_target_loop:
                        # This is a nested loop inside our target loop
                        current_depth += 1
                    continue
                
                # If we're in a target loop section (at depth 0, not in nested loops)
                if in_target_loop and current_depth == 0:
                    # Look for vectorization success markers
                    # Intel: "remark #15300: LOOP WAS VECTORIZED"
                    if re.search(r'remark\s+#15300.*LOOP WAS VECTORIZED', line, re.IGNORECASE):
                        return True  # Found vectorization in one of the versions!
                
                # Track LOOP END to manage nesting
                if line.strip().startswith('LOOP END'):
                    if in_target_loop:
                        if current_depth > 0:
                            current_depth -= 1
                        else:
                            # End of our target loop section
                            in_target_loop = False
        
        # GCC/Clang format (single line with location and status)
        patterns_success = [
            rf'{filename}:{line_number}:\d+:\s*optimized.*vectorized',
            rf'{filename}:{line_number}:\d+:\s*remark.*vectorized',
        ]
        
        for pattern in patterns_success:
            if re.search(pattern, report, re.IGNORECASE):
                # Make sure it's not a negative message
                if not re.search(rf'{filename}:{line_number}.*not vectorized', report, re.IGNORECASE):
                    if not re.search(rf'{filename}:{line_number}.*missed', report, re.IGNORECASE):
                        return True
        
        return False
    
    def verify_file(self, source_file: str, function_name: str = None) -> Dict:
        """
        Verify vectorization for a file across all configured compilers
        
        Args:
            source_file: Source file path
            function_name: Optional function name to filter analysis (default: analyze all functions)
            
        Returns:
            Dictionary with results
        """
        print(f"Verifying file: {source_file}")
        print("=" * 100)
        
        # Step 1: Extract loops and group by nesting structure
        print("\nStep 1: Extracting loops...")
        all_loops = self.loop_analyzer.analyze_file(source_file)
        
        # Group loops by function
        function_loops = {}
        for loop in all_loops:
            func_name = loop.function_name
            if func_name not in function_loops:
                function_loops[func_name] = []
            function_loops[func_name].append(loop)
        
        # For each function, identify nested loop structures
        nested_structures = []
        for func_name, func_loops in function_loops.items():
            # Sort by line number
            func_loops.sort(key=lambda l: l.line_number)
            
            # Group into nested structures based on actual nesting
            # A structure starts with nesting_level 0 and includes all subsequent loops
            # until we return to nesting_level 0 or end of function
            i = 0
            while i < len(func_loops):
                loop = func_loops[i]
                
                # Start of a new structure (nesting level 0)
                if loop.nesting_level == 0:
                    structure = {
                        'function': func_name,
                        'outermost_loop': loop,
                        'all_loops': [loop],
                        'innermost_loop': loop
                    }
                    
                    # Collect all nested loops that follow
                    j = i + 1
                    while j < len(func_loops) and func_loops[j].nesting_level > 0:
                        nested = func_loops[j]
                        structure['all_loops'].append(nested)
                        # Track the deepest nested loop
                        if nested.nesting_level > structure['innermost_loop'].nesting_level:
                            structure['innermost_loop'] = nested
                        j += 1
                    
                    nested_structures.append(structure)
                    i = j  # Skip past the nested loops we just processed
                else:
                    i += 1
        
        # Filter by function name if specified
        if function_name:
            original_count = len(nested_structures)
            available_functions = sorted(set(s['function'] for s in nested_structures))
            filtered_structures = [s for s in nested_structures if s['function'] == function_name]
            
            if len(filtered_structures) == 0:
                print(f"\nError: No loops found in function '{function_name}'")
                if available_functions:
                    print(f"Available functions in file: {', '.join(available_functions)}")
                return {}
            
            nested_structures = filtered_structures
            print(f"\nFiltering: Analyzing only loops in function '{function_name}' ({len(nested_structures)}/{original_count} loop structure(s))")
        
        # Print loop structure information in table format
        filename = os.path.basename(source_file)
        
        # Group structures by function
        function_structures = {}
        for struct in nested_structures:
            func_name = struct['function']
            if func_name not in function_structures:
                function_structures[func_name] = []
            function_structures[func_name].append(struct)
        
        # Print table header
        print()
        print(f"{'File Name':<25} {'Function Name':<20} {'Loop Detected':<65}")
        print("-" * 110)
        
        # Print each function and its loops
        for func_name, structures in function_structures.items():
            first_row = True
            for struct in structures:
                nesting_depth = len(struct['all_loops'])
                innermost_line = struct['innermost_loop'].line_number
                outermost_line = struct['outermost_loop'].line_number
                
                # Only show innermost info for nested loops (2+ levels)
                if nesting_depth > 1:
                    loop_desc = f"{nesting_depth} level nested loop (line #{outermost_line}), innermost at line #{innermost_line}"
                else:
                    loop_desc = f"{nesting_depth} level nested loop (line #{outermost_line})"
                
                if first_row:
                    print(f"{filename:<25} {func_name:<20} {loop_desc:<65}")
                    first_row = False
                else:
                    print(f"{'':<25} {'':<20} {loop_desc:<65}")
            
            # Add separator line after each function
            print("-" * 110)
        
        print(f"\nINFO: Tool will check vectorization of innermost loop in all {len(nested_structures)} loop structures")
        
        if len(nested_structures) == 0:
            return {}
        
        # Create loop dict using innermost loops with their line numbers
        loop_dict = {}
        for struct in nested_structures:
            innermost = struct['innermost_loop']
            func_name = struct['function']
            # Use innermost loop's line number for the loop ID
            loop_id = create_loop_id(source_file, func_name, innermost.line_number)
            
            # Store the complete nested structure for context
            innermost.nested_structure = struct['all_loops']
            
            # Extract complete nested loop code (from outermost to innermost)
            complete_nested_code = self.loop_analyzer.extract_nested_loop_code(
                source_file, struct['all_loops']
            )
            innermost.complete_nested_code = complete_nested_code
            
            loop_dict[loop_id] = innermost
        
        # Step 2: Query LLM for theoretical vectorization of innermost loops
        print("\nStep 2: Querying theoretical vectorization of innermost loop in each structure to LLM with full file context...")
        
        # Read complete file content
        with open(source_file, 'r') as f:
            full_file_content = f.read()
        
        llm_results = self.llm_checker.batch_check(loop_dict, full_file_content=full_file_content)
        
        # Step 3: Verify with each compiler
        print("\nStep 3: Verifying with compilers...")
        compiler_results = {}
        
        for compiler_config in self.compilers:
            print(f"\n  Compiler: {compiler_config.name}")
            
            # Check compiler availability
            available, message = self._check_compiler(compiler_config)
            print(f"    {message}")
            
            if not available:
                print(f"    Skipping {compiler_config.name}")
                continue
            
            # Compile and get report
            success, report = self._compile_and_get_report(source_file, compiler_config)
            
            if not success:
                print(f"    Compilation failed: {report}")
                continue
            
            # Save optimization report for this compiler
            base_name = os.path.splitext(source_file)[0]  # Remove extension
            # Include function name in report filename if specified
            if function_name:
                report_file = f"{base_name}_{function_name}_{compiler_config.name}.optrpt"
            else:
                report_file = f"{base_name}_{compiler_config.name}.optrpt"
            with open(report_file, 'w') as f:
                f.write(f"=== {compiler_config.name.upper()} OPTIMIZATION REPORT ===\n")
                f.write(f"Compiler: {compiler_config.compiler_command}\n")
                f.write(f"Flags: {compiler_config.flags}\n")
                f.write("=" * 80 + "\n\n")
                f.write(report)
                f.write("\n" + "=" * 80 + "\n")
            print(f"    Saved optimization report to: {report_file}")
            
            # Parse report for each loop
            compiler_results[compiler_config.name] = {}
            
            for loop_id, loop in loop_dict.items():
                is_vectorized = self._parse_vectorization_status(
                    report, source_file, loop.line_number, compiler_config.name
                )
                compiler_results[compiler_config.name][loop_id] = is_vectorized
            
            vectorized_count = sum(compiler_results[compiler_config.name].values())
            print(f"    Vectorized: {vectorized_count}/{len(loop_dict)} loops")
        
        # Step 4: Combine results
        combined_results = {}
        
        for loop_id, loop in loop_dict.items():
            llm_result = llm_results.get(loop_id, {})
            
            # Collect compiler results
            compiler_statuses = {}
            for compiler_name in compiler_results.keys():
                compiler_statuses[compiler_name] = compiler_results[compiler_name].get(loop_id, False)
            
            combined_results[loop_id] = {
                'loop': loop,
                'llm_vectorizable': llm_result.get('vectorizable', False),
                'llm_confidence': llm_result.get('confidence', 0.0),
                'llm_reasoning': llm_result.get('reasoning', ''),
                'compiler_results': compiler_statuses
            }
        
        return combined_results
    
    def generate_report(self, results: Dict, output_file: str):
        """Generate text report"""
        lines = []
        
        lines.append("=" * 120)
        lines.append("LLM vs MULTI-COMPILER VECTORIZATION VERIFICATION REPORT")
        lines.append("=" * 120)
        lines.append("")
        
        # Get compiler names
        compiler_names = []
        if results:
            first_result = next(iter(results.values()))
            compiler_names = list(first_result['compiler_results'].keys())
        
        # Summary
        total = len(results)
        llm_yes = sum(1 for r in results.values() if r['llm_vectorizable'])
        
        lines.append(f"Total loops analyzed: {total}")
        lines.append(f"LLM predicted vectorizable: {llm_yes}")
        lines.append("")
        
        for compiler_name in compiler_names:
            compiler_yes = sum(1 for r in results.values() if r['compiler_results'].get(compiler_name, False))
            lines.append(f"{compiler_name.upper()} vectorized: {compiler_yes}")
        
        lines.append("")
        
        # Table header
        header = f"{'LOOP ID':<50} | {'LLM':<15}"
        for compiler_name in compiler_names:
            header += f" | {compiler_name.upper():<10}"
        lines.append("-" * 120)
        lines.append(header)
        lines.append("-" * 120)
        
        # Table rows
        for loop_id, result in results.items():
            llm_status = "YES" if result['llm_vectorizable'] else "NO"
            row = f"{loop_id:<50} | {llm_status:<15}"
            
            for compiler_name in compiler_names:
                status = "VEC" if result['compiler_results'].get(compiler_name, False) else "NO"
                row += f" | {status:<10}"
            
            lines.append(row)
        
        lines.append("-" * 120)
        lines.append("")
        
        # Detailed analysis
        lines.append("DETAILED ANALYSIS")
        lines.append("=" * 120)
        
        for loop_id, result in results.items():
            lines.append("")
            lines.append(f"Loop: {loop_id}")
            lines.append(f"  Line: {result['loop'].line_number}")
            lines.append(f"  Type: {result['loop'].loop_type}")
            lines.append(f"  LLM: {'VECTORIZABLE' if result['llm_vectorizable'] else 'NOT VECTORIZABLE'} ({result['llm_confidence']:.0%})")
            
            for compiler_name in compiler_names:
                status = "VECTORIZED" if result['compiler_results'].get(compiler_name, False) else "NOT VECTORIZED"
                lines.append(f"  {compiler_name.upper()}: {status}")
            
            if result['llm_reasoning'] and not result['llm_reasoning'].startswith('Error:'):
                # Show complete reasoning without truncation
                lines.append(f"  Reason: {result['llm_reasoning']}")
        
        lines.append("")
        lines.append("=" * 120)
        
        report_text = '\n'.join(lines)
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nText report saved to: {output_file}")
        return report_text
    
    def generate_csv_report(self, results: Dict, output_file: str):
        """Generate CSV report (without reason column)"""
        # Get compiler names
        compiler_names = []
        if results:
            first_result = next(iter(results.values()))
            compiler_names = list(first_result['compiler_results'].keys())
        
        with open(output_file, 'w', newline='') as f:
            # Build header (no reason column)
            header = ['LOOP_ID', 'LLM_VECTORIZABLE']
            for compiler_name in compiler_names:
                header.append(f'{compiler_name.upper()}_VECTORIZED')
            
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Write data (no reason)
            for loop_id, result in results.items():
                row = [
                    loop_id,
                    'YES' if result['llm_vectorizable'] else 'NO'
                ]
                
                for compiler_name in compiler_names:
                    status = 'YES' if result['compiler_results'].get(compiler_name, False) else 'NO'
                    row.append(status)
                
                writer.writerow(row)
        
        print(f"CSV report saved to: {output_file}")


def main():
    """Main function"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description='Multi-compiler vectorization verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all loops in a file
  python src/multi_compiler_verifier.py --file code.c
  
  # Analyze only loops in a specific function
  python src/multi_compiler_verifier.py --file code.c --function matrix_multiply
  
  # Use custom config file
  python src/multi_compiler_verifier.py --file code.c --config my_config.yaml
  
Output files are automatically created:
  - filename_YYYYMMDD_HHMMSS.txt (text report)
  - filename_YYYYMMDD_HHMMSS.csv (CSV report)
  - filename_loops_YYYYMMDD_HHMMSS.txt (extracted loops)
  
  With --function flag:
  - filename_functionname_YYYYMMDD_HHMMSS.txt
  - filename_functionname_YYYYMMDD_HHMMSS.csv
  - filename_functionname_loops_YYYYMMDD_HHMMSS.txt
        """
    )
    
    parser.add_argument('--file', required=True,
                       help='C/C++ source file to analyze')
    parser.add_argument('--config', default='config/compiler_config.yaml',
                       help='Path to compiler configuration YAML file')
    parser.add_argument('--function', default=None,
                       help='Filter analysis to loops in specific function (optional)')
    
    args = parser.parse_args()
    
    # Check file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    # Generate output filenames with timestamp
    file_path = Path(args.file)
    file_dir = file_path.parent
    file_stem = file_path.stem  # filename without extension
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Include function name in output filenames if specified
    if args.function:
        base_name = f"{file_stem}_{args.function}"
    else:
        base_name = file_stem
    
    output_txt = file_dir / f"{base_name}_{timestamp}.txt"
    output_csv = file_dir / f"{base_name}_{timestamp}.csv"
    output_loops = file_dir / f"{base_name}_loops_{timestamp}.txt"
    
    print(f"Output files will be created:")
    print(f"  Loops file: {output_loops}")
    print(f"  Text report: {output_txt}")
    print(f"  CSV report: {output_csv}")
    print()
    
    # Create verifier
    try:
        verifier = MultiCompilerVerifier(config_file=args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Verify file
    results = verifier.verify_file(args.file, args.function)
    
    if not results:
        print("No loops found or processed")
        sys.exit(0)
    
    # Save all loops to file
    with open(output_loops, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"ALL LOOPS EXTRACTED FROM: {args.file}\n")
        f.write("=" * 100 + "\n\n")
        
        for loop_id, result in results.items():
            loop = result['loop']
            f.write(f"Loop ID: {loop_id}\n")
            f.write(f"  Innermost Loop Line Number: {loop.line_number}\n")
            f.write(f"  Loop Type: {loop.loop_type}\n")
            f.write(f"  Nesting Level: {loop.nesting_level}\n")
            
            # Show nesting structure if available
            if hasattr(loop, 'nested_structure') and loop.nested_structure:
                f.write(f"  Nested Structure: {len(loop.nested_structure)} level(s)\n")
                for i, nested_loop in enumerate(loop.nested_structure):
                    f.write(f"    Level {i}: {nested_loop.loop_type} at line {nested_loop.line_number}\n")
            
            f.write(f"\nComplete Nested Loop Code:\n")
            f.write("-" * 100 + "\n")
            # Write complete nested loop code if available, otherwise just innermost
            if hasattr(loop, 'complete_nested_code') and loop.complete_nested_code:
                f.write(loop.complete_nested_code)
            else:
                f.write(loop.source_code)
            f.write("\n" + "-" * 100 + "\n\n")
    
    print(f"Loops saved to: {output_loops}")
    
    # Generate reports
    verifier.generate_report(results, str(output_txt))
    verifier.generate_csv_report(results, str(output_csv))
    
    print("\nVerification complete!")
    print(f"  Loops file: {output_loops}")
    print(f"  Text report: {output_txt}")
    print(f"  CSV report: {output_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
