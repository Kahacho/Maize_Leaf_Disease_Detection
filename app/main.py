from fastapi import FastAPI

from .api.classify_image_api import router as classify_image_router


def app() -> FastAPI:
    main_app = FastAPI()
    main_app.include_router(classify_image_router)

    @main_app.get("/")
    async def home():
        return {"Message": "Maize Disease Classifier"}

    return main_app


if __name__ == "__main__":
    app()
