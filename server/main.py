from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "FastAPI", "Status": "Running"}

@app.get("/hello/{name}")
def read_item(name : str):
    return f"HELLO + {name}"