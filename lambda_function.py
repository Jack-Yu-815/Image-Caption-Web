import json
import base64
from pathlib import Path
from eval import infer_from_file, infer_from_image
from io import BytesIO
from PIL import Image


def lambda_handler(event, context):
    encoded_string = event['body']

    image = Image.open(BytesIO(base64.b64decode(encoded_string)))

    # # ChatGPT told me this:
    # image_bytes = event['body']
    # image = Image.open(BytesIO(image_bytes))
    caption = infer_from_image(image)

    print("type(encoded_string):", type(encoded_string))

    # file_path = "__cache__/temp_image"
    # with open(file_path, "wb") as file:
    #     file.write(base64.decodebytes(bytes(encoded_string, 'utf-8')))
    # caption = infer_from_file(file_path)
    # print(caption)
    #
    # # delete the temp file
    # p = Path(file_path)
    # if p.exists() and p.is_file():
    #     p.unlink()

    response_dict = {
        "caption": caption
    }

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST'
        },
        'body': json.dumps(response_dict)
    }
