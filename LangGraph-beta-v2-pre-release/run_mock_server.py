import os

import uvicorn

os.environ["LLM_PROVIDER"] = "mock"

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8080)

