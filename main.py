import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    # Listen to the environment PORT of the cloud server
    server_port = int(os.environ.get('PORT', 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=server_port, log_level="info")
