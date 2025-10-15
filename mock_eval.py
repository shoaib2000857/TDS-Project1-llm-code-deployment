from fastapi import FastAPI, Request
app = FastAPI()

@app.post("/notify")
async def notify(req: Request):
    data = await req.json()
    print("✅ got eval payload:", data)
    return {"ok": True}
