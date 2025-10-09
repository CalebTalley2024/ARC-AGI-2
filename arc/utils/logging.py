"""
Step 9: Logging Utilities

This module provides structured logging for the ARC project.

Goal: Create consistent, informative logs for debugging and monitoring.

Key concepts:
- Structured logging: JSON-formatted logs with context
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Rich formatting: Colored output for terminal
- File logging: Persistent logs for analysis

Recommended libraries:
- loguru: Simple, powerful logging with great defaults
- rich: Beautiful terminal output with colors and formatting
- Standard logging: Built-in Python logging module

Functions to implement:

1. setup_logger(name: str = "arc", level: str = "INFO", log_file: str = None) -> Logger
   # Initialize logger with consistent formatting
   # Args:
   #   name: Logger name (usually module name)
   #   level: Minimum log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
   #   log_file: Optional file path for persistent logs
   # Returns: Configured logger object
   #
   # Configuration:
   #   - Console handler with rich formatting
   #   - File handler with JSON formatting (if log_file provided)
   #   - Include timestamp, level, module, function, line number
   #   - Add context fields (task_id, view_spec, etc.)

2. log_task_start(logger, task_id: str, n_train: int, n_test: int) -> None
   # Log start of task processing
   # Format: "Starting task abc123 | train=3 test=1"

3. log_task_complete(logger, task_id: str, time_elapsed: float, success: bool) -> None
   # Log task completion
   # Format: "Completed task abc123 | 2.34s | success=True"

4. log_view_generation(logger, task_id: str, n_views: int, time_elapsed: float) -> None
   # Log view generation statistics
   # Format: "Generated 48 views for task abc123 | 0.12s"

5. log_model_inference(logger, task_id: str, view_spec: str, time_elapsed: float) -> None
   # Log model inference timing
   # Format: "Inference for view rot90|0123456789|row | 0.45s"

6. log_error(logger, task_id: str, error: Exception, context: dict = None) -> None
   # Log error with full context
   # Include: task_id, error type, message, traceback, context

7. create_progress_bar(total: int, desc: str = "Processing") -> ProgressBar
   # Create rich progress bar for iterations
   # Args:
   #   total: Total number of items
   #   desc: Description text
   # Returns: Progress bar object (tqdm or rich.progress)

Log format examples:

Console (rich formatting):
  [2025-10-08 23:11:52] INFO     Starting task abc123 | train=3 test=1
  [2025-10-08 23:11:54] DEBUG    Generated 48 views | 0.12s
  [2025-10-08 23:11:55] INFO     Completed task abc123 | 2.34s ✓

File (JSON):
  {"timestamp": "2025-10-08T23:11:52", "level": "INFO", "module": "runner",
   "function": "solve_task", "line": 42, "message": "Starting task abc123",
   "context": {"task_id": "abc123", "n_train": 3, "n_test": 1}}

Best practices:
- Log at appropriate levels (don't spam DEBUG in production)
- Include context (task_id, view_spec, etc.)
- Log timing for performance analysis
- Log errors with full traceback
- Use structured logging (JSON) for parsing
- Rotate log files to prevent disk fill
- Don't log sensitive data

Common pitfalls:
- Logging too much (slows down execution)
- Logging too little (hard to debug)
- Not including context (which task failed?)
- Logging in tight loops (use sampling)
- Not rotating log files (disk fills up)
"""

# TODO: Import logging
# TODO: Import loguru (or use standard logging)
# TODO: Import rich (Console, Progress)
# TODO: Import json
# TODO: Import datetime

# TODO: Implement setup_logger(name, level, log_file) -> Logger
#   - Configure console handler with rich formatting
#   - Configure file handler with JSON formatting (if log_file)
#   - Set log level
#   - Return logger

# TODO: Implement log_task_start(logger, task_id, n_train, n_test) -> None
#   - Log INFO message with task details

# TODO: Implement log_task_complete(logger, task_id, time_elapsed, success) -> None
#   - Log INFO message with timing and success status

# TODO: Implement log_view_generation(logger, task_id, n_views, time_elapsed) -> None
#   - Log DEBUG message with view count and timing

# TODO: Implement log_model_inference(logger, task_id, view_spec, time_elapsed) -> None
#   - Log DEBUG message with view spec and timing

# TODO: Implement log_error(logger, task_id, error, context) -> None
#   - Log ERROR message with full traceback and context

# TODO: Implement create_progress_bar(total, desc) -> ProgressBar
#   - Use tqdm or rich.progress
#   - Return configured progress bar
