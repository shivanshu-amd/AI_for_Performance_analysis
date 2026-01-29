"""
LLM-Based Vectorization Checker
Uses LLM (Gemini or O3-mini) to verify loop vectorization potential
"""

import openai
import os
import sys
import re
from typing import Dict, Tuple

# Add parent src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from loop_analyzer import LoopInfo


class LLMVectorizationChecker:
    """
    Uses on-prem LLM to check if a loop is vectorizable
    """
    
    def __init__(self, model='default', api_key=None):
        """
        Initialize LLM checker with on-prem endpoint
        
        Args:
            model: Model name (default: 'default' for on-prem)
            api_key: API key (default: from environment or hardcoded)
        """
        self.model = model
        self.api_key = api_key or '8520f49d6c804008beeafd11288ff507'
        
        # Setup on-prem client
        self.client = self._setup_onprem()
    
    def _setup_onprem(self):
        """Setup On-Prem client"""
        url = 'https://llm-api.amd.com'
        
        # Get username (handle WSL/Linux issues)
        try:
            username = os.getlogin()
        except:
            username = os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            "user": username
        }
        model_api_version = '2024-12-01-preview'
        
        # On-prem uses 'default' as deployment ID
        deployment_id = 'default'
        
        client = openai.AzureOpenAI(
            api_key='dummy',
            api_version=model_api_version,
            base_url=url,
            default_headers=headers
        )
        # Use on-prem endpoint: /api/OnPrem/openai/deployments/default
        client.base_url = f'{url}/api/OnPrem/openai/deployments/{deployment_id}'
        
        return client
    
    def check_vectorization(self, loop_code: str, loop_id: str, full_file_content: str = None, line_number: int = None) -> Tuple[bool, str, float]:
        """
        Ask LLM if loop is vectorizable
        
        Args:
            loop_code: The loop source code
            loop_id: Loop identifier (filename_functionname_line#)
            full_file_content: Complete file content for context (optional)
            line_number: Line number of the loop (optional)
            
        Returns:
            is_vectorizable: True if LLM says vectorizable
            reasoning: LLM's explanation
            confidence: Confidence score (0-1)
        """
        # Create prompt for LLM with full context
        if full_file_content and line_number:
            prompt = f"""You are an expert compiler optimization engineer specializing in vectorization analysis.

Analyze the INNERMOST loop (at line {line_number}) in this nested loop structure and determine if it can be vectorized by a modern compiler (GCC, Clang, Intel).

Complete File for Context:
```c
{full_file_content}
```

Loop Structure to Analyze (focus on innermost loop at line {line_number}):
```c
{loop_code}
```

Please analyze whether the INNERMOST loop at line {line_number} can be vectorized and respond in this exact format:
LOOP_ID: {loop_id}
VECTORIZABLE: [YES/NO]
CONFIDENCE: [0-100]
REASONING: [Your complete detailed technical explanation - do not truncate]

As an expert, consider these factors with full file context:
1. Variable types and declarations (check data types, alignment)
2. Array access patterns (unit stride is optimal for SIMD)
3. Data dependencies (loop-carried dependencies prevent vectorization)
4. Function calls (prevent vectorization unless inlined or vectorizable)
5. Conditionals (may prevent or complicate vectorization)
6. Memory access patterns (gather/scatter vs contiguous)
7. Control flow (break/continue/return prevent vectorization)
8. Reduction patterns (sum, product - can be vectorized with special techniques)
9. Outer loop effects on inner loop vectorization

Provide complete technical reasoning about the INNERMOST loop's vectorization potential without truncation."""
        else:
            # Fallback to loop-only analysis
            prompt = f"""Analyze this C/C++ loop and determine if it can be vectorized by a compiler.

Loop Code:
```c
{loop_code}
```

Please analyze and respond in this exact format:
VECTORIZABLE: [YES/NO]
CONFIDENCE: [0-100]
REASONING: [Your detailed explanation]

Consider these factors:
1. Array access patterns (unit stride is best)
2. Data dependencies (loop-carried dependencies prevent vectorization)
3. Function calls (prevent vectorization unless inlined)
4. Conditionals (complicate vectorization)
5. Memory access patterns
6. Control flow (break/continue/return prevent vectorization)

Be concise and technical in your reasoning."""

        try:
            # Query on-prem LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                stream=False,
                max_completion_tokens=1024
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract vectorizable status
            vectorizable_match = re.search(r'VECTORIZABLE:\s*(YES|NO)', content, re.IGNORECASE)
            is_vectorizable = vectorizable_match.group(1).upper() == 'YES' if vectorizable_match else False
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', content)
            confidence = float(confidence_match.group(1)) / 100.0 if confidence_match else 0.5
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else content
            
            return is_vectorizable, reasoning, confidence
            
        except Exception as e:
            print(f"Error querying LLM for {loop_id}: {e}")
            return False, f"Error: {str(e)}", 0.0
    
    def batch_check(self, loops: Dict[str, LoopInfo], full_file_content: str = None) -> Dict[str, Dict]:
        """
        Check multiple loops
        
        Args:
            loops: Dictionary mapping loop_id to LoopInfo
            full_file_content: Complete file content for context (optional)
            
        Returns:
            Dictionary mapping loop_id to results
        """
        results = {}
        
        total = len(loops)
        for i, (loop_id, loop) in enumerate(loops.items(), 1):
            print(f"[{i}/{total}] Checking {loop_id}...")
            
            # Use complete nested loop code if available, otherwise use innermost loop code
            loop_code_to_analyze = loop.source_code
            if hasattr(loop, 'complete_nested_code') and loop.complete_nested_code:
                loop_code_to_analyze = loop.complete_nested_code
            
            is_vectorizable, reasoning, confidence = self.check_vectorization(
                loop_code_to_analyze,
                loop_id,
                full_file_content=full_file_content,
                line_number=loop.line_number
            )
            
            results[loop_id] = {
                'vectorizable': is_vectorizable,
                'reasoning': reasoning,
                'confidence': confidence,
                'loop': loop
            }
        
        return results


def create_loop_id(filename: str, function_name: str, line_number: int) -> str:
    """
    Create unique loop identifier
    
    Args:
        filename: Source filename
        function_name: Function containing the loop
        line_number: Line number of loop
        
    Returns:
        Loop ID in format: filename_functionname_line#
    """
    # Clean filename (remove path and extension)
    clean_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # Clean function name (remove special characters)
    clean_function = re.sub(r'[^a-zA-Z0-9_]', '', function_name)
    
    return f"{clean_filename}_{clean_function}_line{line_number}"


if __name__ == "__main__":
    print("LLM Vectorization Checker")
    print("=" * 80)
    print()
    print("This module uses LLM to check loop vectorization potential.")
    print()
    print("Supported models:")
    print("  - gemini-3-pro (Google Gemini)")
    print("  - o3-mini (OpenAI O3)")
    print()
    print("Usage:")
    print("  from src_LLM.llm_vectorization_checker import LLMVectorizationChecker")
    print()
    print("  checker = LLMVectorizationChecker(model='gemini-3-pro')")
    print("  is_vec, reasoning, conf = checker.check_vectorization(code, loop_id)")
