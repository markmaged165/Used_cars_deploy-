from fastapi import FastAPI

app = FastAPI()

# Define an endpoint BEFORE requesting it
@app.get("/hello")
def say_hello():
    return {"message": "Hello World"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
