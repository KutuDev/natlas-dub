
import os
import uvicorn
from run_project import initialize

app = initialize()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
