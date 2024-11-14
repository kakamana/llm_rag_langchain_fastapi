import modal

app=modal.App("example-get-started")

@app.function()
def square(x):
    print ("This code is running on a remote server work load!")
    return x**2

@app.local_entrypoint()
def main():
    print ("This code is running on the local machine!")
    print("the squar of 42 is", square.remote(42))