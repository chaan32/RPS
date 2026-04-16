import os
import boto3
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("S3_BUCKET")
REGION = os.getenv("AWS_REGION", "ap-northeast-2")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=REGION,
)


def upload_file(file_bytes: bytes, key: str, content_type: str = "image/jpeg") -> str:
    """
    바이트 데이터를 S3에 업로드하고 public URL 반환
    key: S3 내부 경로 (예: 'snapshots/2026-04-15/abc.jpg')
    """
    s3_client.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )
    return f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{key}"
