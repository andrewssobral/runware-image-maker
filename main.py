import fastapi
import uvicorn

from image_maker import router


def main() -> None:
    app: fastapi.FastAPI = fastapi.FastAPI()
    app.include_router(router.ROUTER)
    uvicorn.run(app, port=12345)


if __name__ == "__main__":
    main()
