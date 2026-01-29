"""
Loop Analyzer Module
Detects and analyzes loops in C/C++ source code
"""

import re
from typing import List, Dict, Tuple
from pycparser import c_parser, c_ast, parse_file


class LoopInfo:
    """Represents information about a loop"""
    
    def __init__(self, loop_type: str, line_number: int, source_code: str):
        self.loop_type = loop_type  # 'for', 'while', 'do-while'
        self.line_number = line_number
        self.source_code = source_code
        self.nesting_level = 0
        self.body_statements = []
        self.init_expr = None
        self.condition = None
        self.increment = None
        self.function_name = "unknown"  # Function containing this loop
        
    def __repr__(self):
        return f"Loop({self.loop_type}, line={self.line_number}, func={self.function_name}, nesting={self.nesting_level})"


class LoopVisitor(c_ast.NodeVisitor):
    """AST visitor to find and analyze loops"""
    
    def __init__(self):
        self.loops = []
        self.current_nesting = 0
        self.current_function = "unknown"  # Track current function name
    
    def visit_FuncDef(self, node):
        """Visit function definitions to track current function"""
        # Save previous function name
        prev_function = self.current_function
        
        # Set current function name
        if node.decl and node.decl.name:
            self.current_function = node.decl.name
        
        # Visit children (including loops in function body)
        self.generic_visit(node)
        
        # Restore previous function name (for nested functions, if any)
        self.current_function = prev_function
        
    def visit_For(self, node):
        """Visit for loops"""
        loop = LoopInfo('for', node.coord.line if node.coord else 0, '')
        loop.nesting_level = self.current_nesting
        loop.function_name = self.current_function  # Set function name
        
        # Extract loop components
        if node.init:
            loop.init_expr = self._node_to_string(node.init)
        if node.cond:
            loop.condition = self._node_to_string(node.cond)
        if node.next:
            loop.increment = self._node_to_string(node.next)
            
        self.loops.append(loop)
        
        # Visit nested loops
        self.current_nesting += 1
        self.generic_visit(node)
        self.current_nesting -= 1
        
    def visit_While(self, node):
        """Visit while loops"""
        loop = LoopInfo('while', node.coord.line if node.coord else 0, '')
        loop.nesting_level = self.current_nesting
        loop.function_name = self.current_function  # Set function name
        
        if node.cond:
            loop.condition = self._node_to_string(node.cond)
            
        self.loops.append(loop)
        
        self.current_nesting += 1
        self.generic_visit(node)
        self.current_nesting -= 1
        
    def visit_DoWhile(self, node):
        """Visit do-while loops"""
        loop = LoopInfo('do-while', node.coord.line if node.coord else 0, '')
        loop.nesting_level = self.current_nesting
        loop.function_name = self.current_function  # Set function name
        
        if node.cond:
            loop.condition = self._node_to_string(node.cond)
            
        self.loops.append(loop)
        
        self.current_nesting += 1
        self.generic_visit(node)
        self.current_nesting -= 1
        
    def _node_to_string(self, node):
        """Convert AST node to string representation"""
        if node is None:
            return ""
        # Simple string representation
        return str(node)


class LoopAnalyzer:
    """Main class for analyzing loops in C/C++ code"""
    
    def __init__(self):
        self.parser = c_parser.CParser()
        
    def analyze_file(self, filepath: str, use_cpp: bool = False) -> List[LoopInfo]:
        """
        Analyze a C/C++ file and extract all loops
        
        Args:
            filepath: Path to the source file
            use_cpp: Whether to use C preprocessor
            
        Returns:
            List of LoopInfo objects
        """
        try:
            # Parse the file
            if use_cpp:
                ast = parse_file(filepath, use_cpp=True)
            else:
                with open(filepath, 'r') as f:
                    code = f.read()
                ast = self.parser.parse(code, filepath)
            
            # Visit AST and collect loops
            visitor = LoopVisitor()
            visitor.visit(ast)
            
            # Enhance loop information with source code
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            for loop in visitor.loops:
                if loop.line_number > 0 and loop.line_number <= len(lines):
                    loop.source_code = self._extract_loop_code(lines, loop.line_number)
                    
            return visitor.loops
            
        except Exception as e:
            # Silently fall back to regex-based analysis
            # (AST parser doesn't support comments, but regex parser handles them fine)
            return self._regex_based_analysis(filepath)
    
    def _extract_loop_code(self, lines: List[str], start_line: int) -> str:
        """Extract the complete loop code including body by matching braces"""
        if start_line < 1 or start_line > len(lines):
            return ""
        
        # Start from the loop line
        result = []
        brace_count = 0
        started = False
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            result.append(line)
            
            # Count braces
            for char in line:
                if char == '{':
                    brace_count += 1
                    started = True
                elif char == '}':
                    brace_count -= 1
            
            # If we've matched all braces, we're done
            if started and brace_count == 0:
                break
            
            # Safety limit - don't extract more than 100 lines
            if len(result) >= 100:
                break
        
        return ''.join(result)
    
    def extract_nested_loop_code(self, filepath: str, nested_loops: List['LoopInfo']) -> str:
        """
        Extract complete nested loop structure code
        
        Args:
            filepath: Source file path
            nested_loops: List of LoopInfo objects in the nested structure (outermost to innermost)
            
        Returns:
            Complete nested loop code as string
        """
        if not nested_loops:
            return ""
        
        # Read file lines
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Get the outermost loop's starting line
        outermost = nested_loops[0]
        start_line = outermost.line_number
        
        # Extract from outermost loop start to where all braces close
        return self._extract_loop_code(lines, start_line)
    
    def _regex_based_analysis(self, filepath: str) -> List[LoopInfo]:
        """
        Fallback regex-based loop detection when AST parsing fails
        Also extracts function names and complete loop bodies
        """
        loops = []
        
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Pattern for for loops
        for_pattern = r'^\s*for\s*\('
        # Pattern for while loops
        while_pattern = r'^\s*while\s*\('
        # Pattern for do-while loops
        do_pattern = r'^\s*do\s*\{'
        
        nesting_stack = []
        current_function = "unknown"
        function_brace_depth = 0
        in_function = False
        
        for i, line in enumerate(lines, 1):
            # Look for function definition by checking if line contains opening brace
            # and previous lines contain function signature  
            if '{' in line and not in_function:
                # Look back up to 3 lines for function name
                for j in range(max(0, i-4), i):
                    # Match: word followed by opening parenthesis
                    match = re.search(r'\b(\w+)\s*\(', lines[j])
                    if match:
                        # This is likely the function name
                        potential_name = match.group(1)
                        # Filter out keywords and common types
                        if potential_name not in ['if', 'while', 'for', 'switch', 'sizeof', 'int', 'float', 'double', 'char', 'void']:
                            current_function = potential_name
                            in_function = True
                            function_brace_depth = 0  # Will be incremented below
                            break
            
            # Track brace depth when in function
            if in_function:
                function_brace_depth += line.count('{') - line.count('}')
                
                # Exit function when braces balance
                if function_brace_depth <= 0:
                    in_function = False
                    current_function = "unknown"
                    function_brace_depth = 0
            
            # Check for loop starts
            if re.search(for_pattern, line):
                loop = LoopInfo('for', i, self._extract_loop_code(lines, i))
                loop.nesting_level = len(nesting_stack)
                loop.function_name = current_function
                loops.append(loop)
                nesting_stack.append(loop)
                
            elif re.search(while_pattern, line):
                loop = LoopInfo('while', i, self._extract_loop_code(lines, i))
                loop.nesting_level = len(nesting_stack)
                loop.function_name = current_function
                loops.append(loop)
                nesting_stack.append(loop)
                
            elif re.search(do_pattern, line):
                loop = LoopInfo('do-while', i, self._extract_loop_code(lines, i))
                loop.nesting_level = len(nesting_stack)
                loop.function_name = current_function
                loops.append(loop)
                nesting_stack.append(loop)
            
            # Track loop nesting with braces
            if '}' in line and nesting_stack:
                # Check if this closes a loop
                nesting_stack.pop()
        
        return loops
    
    def analyze_code_string(self, code: str) -> List[LoopInfo]:
        """
        Analyze C/C++ code from a string
        
        Args:
            code: Source code as string
            
        Returns:
            List of LoopInfo objects
        """
        try:
            ast = self.parser.parse(code)
            visitor = LoopVisitor()
            visitor.visit(ast)
            return visitor.loops
        except Exception as e:
            print(f"Error parsing code: {e}")
            return []
    
    def get_loop_statistics(self, loops: List[LoopInfo]) -> Dict:
        """
        Get statistics about the loops found
        
        Args:
            loops: List of LoopInfo objects
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_loops': len(loops),
            'for_loops': sum(1 for l in loops if l.loop_type == 'for'),
            'while_loops': sum(1 for l in loops if l.loop_type == 'while'),
            'do_while_loops': sum(1 for l in loops if l.loop_type == 'do-while'),
            'max_nesting': max([l.nesting_level for l in loops], default=0),
            'nested_loops': sum(1 for l in loops if l.nesting_level > 0),
        }
        return stats


if __name__ == "__main__":
    # Test the analyzer
    analyzer = LoopAnalyzer()
    
    # Example test code
    test_code = """
    int main() {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                printf("%d ", i * j);
            }
        }
        
        int k = 0;
        while (k < 5) {
            k++;
        }
        
        return 0;
    }
    """
    
    loops = analyzer.analyze_code_string(test_code)
    print(f"Found {len(loops)} loops:")
    for loop in loops:
        print(f"  {loop}")
    
    stats = analyzer.get_loop_statistics(loops)
    print(f"\nStatistics: {stats}")
