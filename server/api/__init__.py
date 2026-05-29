"""REST API 라우터 (Spring 의 @RestController 에 해당).

사용:
    from .api import (
        health, audio, alerts, workers, makers,
        images, incidents, reports, aruco,
    )
    app.include_router(health.router)
    ...
"""
