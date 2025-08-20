import importlib
import json
import os
import re
import signal
from datetime import datetime
from typing import Any, Dict, List, Optional

from octotools.engine.factory import create_llm_engine
from octotools.models.formatters import ToolCommand

try:
    TimeoutError
except NameError:
    class TimeoutError(Exception):
        pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

class Executor:
    def __init__(self, llm_engine_name: str, root_cache_dir: str = "solver_cache",  num_threads: int = 1, max_time: int = 120, 
    max_output_length: int = 100000, verbose: bool = False, base_url: str = None, check_model: bool = True, temperature: float = .0):
        self.llm_engine_name = llm_engine_name
        self.root_cache_dir = root_cache_dir
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.verbose = verbose
        self.base_url = base_url
        self.check_model = check_model
        self.temperature  = temperature
    def set_query_cache_dir(self, query_cache_dir):
        if query_cache_dir:
            self.query_cache_dir = query_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.root_cache_dir, timestamp)
        os.makedirs(self.query_cache_dir, exist_ok=True)

    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], step_count: int = 0, json_data: Any = None) -> Any:
#         prompt_generate_tool_command = f"""
# Task: Generate a precise command to execute the selected tool based on the given information.

# Query: {question}
# Image: {image}
# Context: {context}
# Sub-Goal: {sub_goal}
# Selected Tool: {tool_name}
# Tool Metadata: {tool_metadata}

# Instructions:
# 1. Carefully review all provided information: the query, image path, context, sub-goal, selected tool, and tool metadata.
# 2. Analyze the tool's input_types from the metadata to understand required and optional parameters.
# 3. Construct a command or series of commands that aligns with the tool's usage pattern and addresses the sub-goal.
# 4. Ensure all required parameters are included and properly formatted.
# 5. Use appropriate values for parameters based on the given context, particularly the `Context` field which may contain relevant information from previous steps.
# 6. If multiple steps are needed to prepare data for the tool, include them in the command construction.

# Output Format:
# Provide your response in the following structure:

# Analysis: <analysis>
# Command Explanation: <explanation>
# Generated Command:
# ```python
# <command>
# ```

# Where:
# - <analysis> is a step-by-step analysis of the context, sub-goal, and selected tool to guide the command construction.
# - <explanation> is a detailed explanation of the constructed command(s) and their parameters.
# - <command> is the Python code to execute the tool, which can be one of the following types:
#     a. A single line command with `execution = tool.execute()`.
#     b. A multi-line command with complex data preparation, ending with `execution = tool.execute()`.
#     c. Multiple lines of `execution = tool.execute()` calls for processing multiple items.

# Rules:
# 1. The command MUST be valid Python code and include at least one call to `tool.execute()`.
# 2. Each `tool.execute()` call MUST be assigned to the 'execution' variable in the format `execution = tool.execute(...)`.
# 3. For multiple executions, use separate `execution = tool.execute()` calls for each execution.
# 4. The final output MUST be assigned to the 'execution' variable, either directly from `tool.execute()` or as a processed form of multiple executions.
# 5. Use the exact parameter names as specified in the tool's input_types.
# 6. Enclose string values in quotes, use appropriate data types for other values (e.g., lists, numbers).
# 7. Do not include any code or text that is not part of the actual command.
# 8. Ensure the command directly addresses the sub-goal and query.
# 9. Include ALL required parameters, data, and paths to execute the tool in the command itself.
# 10. If preparation steps are needed, include them as separate Python statements before the `tool.execute()` calls.

# Examples (Not to use directly unless relevant):

# Example 1 (Single line command):
# Analysis: The tool requires an image path and a list of labels for object detection.
# Command Explanation: We pass the image path and a list containing "baseball" as the label to detect.
# Generated Command:
# ```python
# execution = tool.execute(image="path/to/image", labels=["baseball"])
# ```

# Example 2 (Multi-line command with data preparation):
# Analysis: The tool requires an image path, multiple labels, and a threshold for object detection.
# Command Explanation: We prepare the data by defining variables for the image path, labels, and threshold, then pass these to the tool.execute() function.
# Generated Command:
# ```python
# image = "path/to/image"
# labels = ["baseball", "football", "basketball"]
# threshold = 0.5
# execution = tool.execute(image=image, labels=labels, threshold=threshold)
# ```

# Example 3 (Multiple executions):
# Analysis: We need to process multiple images for baseball detection.
# Command Explanation: We call the tool for each image path, using the same label and threshold for all.
# Generated Command:
# ```python
# execution = tool.execute(image="path/to/image1", labels=["baseball"], threshold=0.5)
# execution = tool.execute(image="path/to/image2", labels=["baseball"], threshold=0.5)
# execution = tool.execute(image="path/to/image3", labels=["baseball"], threshold=0.5)
# ```

# Some Wrong Examples:
# Generated Command:
# ```python
# execution1 = tool.execute(query="...")
# execution2 = tool.execute(query="...")
# ```
# Reason: only `execution = tool.execute` is allowed, not `execution1` or `execution2`.

# Generated Command:
# ```python
# urls = [
#     "https://example.com/article1",
#     "https://example.com/article2"
# ]

# execution = tool.execute(url=urls[0])
# execution = tool.execute(url=urls[1])
# ```
# Reason: The command should process multiple items in a single execution, not separate executions for each item.

# Remember: Your response MUST end with the Generated Command, which should be valid Python code including any necessary data preparation steps and one or more `execution = tool.execute(` calls, without any additional explanatory text. The format `execution = tool.execute` must be strictly followed, and the last line must begin with `execution = tool.execute` to capture the final output."""
        prompt_generate_tool_command = f"""
        Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {question}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool_name}
- **Tool Metadata:** {tool_metadata}
- **Relevant Data:** {context}

Instructions:
1.  Analyze the tool's required parameters from its metadata.
2.  Construct valid Python code that addresses the sub-goal using the provided context and data.
3.  The command must include at least one call to `tool.execute()`.
4.  Each `tool.execute()` call must be assigned to a variable named **`execution`**.
5.  Please give the exact numbers and parameters should be used in the `tool.execute()` call.

Output Format:
Present your response in the following structured format. Do not include any extra text or explanations.

Generated Command:
```python
<command>
```

Example1:
Generated Command:
```python
execution = tool.execute(prompt="Summarize the following porblom:"Isaac has 100 toys, masa gets ...., how much are their together?")
```

Example2:
Generated Command:
```python
execution = tool.execute(prompt="Solve the following equation when a = 90, b = 9000")
```
"""

        llm_generate_tool_command = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False, base_url=self.base_url, check_model=self.check_model, temperature = self.temperature)
        tool_command = llm_generate_tool_command(prompt_generate_tool_command, response_format=ToolCommand)
        if json_data is not None:
            json_data[f"tool_commander_{step_count}_prompt"] = prompt_generate_tool_command
            json_data[f"tool_commander_{step_count}_response"] = str(tool_command)

        return tool_command

    def extract_explanation_and_command(self, response: Any) -> tuple:
        def normalize_code(code: str) -> str:
            # Remove leading/trailing whitespace and triple backticks if present
            return re.sub(r'^```python\s*', '', code).rstrip('```').strip()

        analysis = "No analysis found."
        explanation = "No explanation found."
        command = "No command found."

        if isinstance(response, str):
            # Attempt to parse as JSON first
            try:
                response_dict = json.loads(response)
                response_obj = ToolCommand(**response_dict)
                analysis = response_obj.analysis.strip()
                explanation = response_obj.explanation.strip()
                command = response_obj.command.strip()
            except Exception as e:
                print(f"Failed to parse response as JSON: {str(e)}")
                # Fall back to regex parsing on string
                try:
                    # Extract analysis
                    analysis_pattern = r"Analysis:(.*?)Command Explanation"
                    analysis_match = re.search(analysis_pattern, response, re.DOTALL | re.IGNORECASE)
                    analysis = analysis_match.group(1).strip() if analysis_match else "No analysis found."

                    # Extract explanation
                    explanation_pattern = r"Command Explanation:(.*?)Generated Command"
                    explanation_match = re.search(explanation_pattern, response, re.DOTALL | re.IGNORECASE)
                    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found."

                    # Extract command using "Generated Command:" prefix
                    command_pattern = r"Generated Command:.*?```python\n(.*?)```"
                    command_match = re.search(command_pattern, response, re.DOTALL | re.IGNORECASE)
                    if command_match:
                        command = command_match.group(1).strip()
                    else:
                        # Fallback: Extract ANY ```python ... ``` block (even without prefix)
                        loose_command_pattern = r"```python\s*\n(.*?)```"
                        loose_match = re.findall(loose_command_pattern, response, re.DOTALL | re.IGNORECASE)
                        if loose_match:
                            # Take the last or most complete one? Or first meaningful?
                            # Here we take the longest one as heuristic
                            command = max(loose_match, key=lambda x: len(x.strip())).strip()
                        else:
                            command = "No command found."
                except Exception as e:
                    print(f"Error during regex parsing: {str(e)}")
                    analysis = "Parsing error."
                    explanation = "Parsing error."
                    command = "No command found."
        elif isinstance(response, ToolCommand):
            analysis = response.analysis.strip()
            explanation = response.explanation.strip()
            command = response.command.strip()
        else:
            command = "Invalid response type."

        # Final normalization
        command = normalize_code(command)

        return analysis, explanation, command

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        """
        Execute a tool command with timeout protection. If execution exceeds max_time seconds,
        the function will be interrupted and return a timeout message.

        Args:
            tool_name (str): Name of the tool to execute
            command (str): Command string containing tool.execute() calls

        Returns:
            Any: List of execution results or error message
        """

        def split_commands(command: str) -> List[str]:
            # Use regex to find all tool.execute() commands and their surrounding code
            pattern = r'.*?execution\s*=\s*tool\.execute\([^\n]*\)\s*(?:\n|$)'
            blocks = re.findall(pattern, command, re.DOTALL)
            return [block.strip() for block in blocks if block.strip()]

        def execute_with_timeout(block: str, local_context: dict) -> Optional[str]:
            # Set up the timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.max_time)

            try:
                # Execute the block in the local context
                exec(block, globals(), local_context)
                result = local_context.get('execution')
                signal.alarm(0)  # Disable the alarm
                return result
            except TimeoutError:
                return f"Execution timed out after {self.max_time} seconds"
            finally:
                signal.alarm(0)  # Ensure alarm is disabled even if other exceptions occur

        # Import the tool module and instantiate it
        module_name = f"tools.{tool_name.lower().replace('_tool', '')}.tool"

        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)

            # Get the tool class
            tool_class = getattr(module, tool_name)

            # Check if the tool requires an LLM engine
            # NOTE may need to refine base.py and tool.py to handle this better
            # if getattr(tool_class, 'require_llm_engine', False):
            #     # Instantiate the tool with the model_string
            #     tool = tool_class(model_string=self.llm_engine_name)
            # else:
            #     # Instantiate the tool without model_string for tools that don't require it
            #     tool = tool_class()
            
            tool = tool_class()

            # Set the custom output directory
            # NOTE: May have a better way to handle this
            tool.set_custom_output_dir(self.query_cache_dir)

            # Split the command into blocks, execute each one and store execution results
            command_blocks = split_commands(command)
            executions = []

            for block in command_blocks:
                # Create a local context to safely execute the block
                local_context = {'tool': tool}

                # Execute the block with timeout protection
                result = execute_with_timeout(block, local_context)

                if result is not None:
                    executions.append(result)
                else:
                    executions.append(f"No execution captured from block: {block}")

            # Return all the execution results
            return executions
        except Exception as e:
            return f"Error in execute_tool_command: {str(e)}"
