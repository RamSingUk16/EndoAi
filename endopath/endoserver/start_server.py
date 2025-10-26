#!/usr/bin/env python3
"""
Startup script for the EndoAI server with proper static file configuration.
"""
import os
import sys
import uvicorn

# Set the static directory to the frontend files
os.environ["STATIC_DIR"] = r"c:\dev\endoui\endopath\endoui"

# Print confirmation
print(f"Setting STATIC_DIR to: {os.environ['STATIC_DIR']}")

# Verify the directory exists
if os.path.exists(os.environ['STATIC_DIR']):
    print(f"✓ Static directory exists with {len(os.listdir(os.environ['STATIC_DIR']))} files")
else:
    print(f"✗ Static directory does not exist: {os.environ['STATIC_DIR']}")
    sys.exit(1)

if __name__ == "__main__":
    # Start the uvicorn server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )