# chat/consumers.py
import json

from channels.generic.websocket import AsyncWebsocketConsumer
from transformers import T5Tokenizer, T5ForConditionalGeneration

from transformers import ViTImageProcessor, ViTForImageClassification

import io, base64
from PIL import Image

class ChatConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vision_model = ViTForImageClassification.from_pretrained('vit-snack')

        self.labels = ['apple', 'banana', 'cake', 'candy', 'carrot', 'cookie', 'doughnut', 'grape', 'hot dog', 'ice cream', 'juice', 'muffin', 'orange', 'pineapple', 'popcorn', 'pretzel', 'salad', 'strawberry', 'waffle', 'watermelon']

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        
        if (str(text_data).startswith("data:")):

            print(text_data[:100])
            img = Image.open(io.BytesIO(base64.decodebytes(bytes(str(text_data).split(",")[-1], "utf-8"))))
            prepraed_img = self.processor(images=img, return_tensors="pt")
            outputs = self.vision_model(**prepraed_img)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            await self.send(text_data=json.dumps([
                {
                    "type": False,
                    "value": f"this is {self.labels[predicted_class_idx]}"
                }
            ]))
        else:
            text_data_json = json.loads(text_data)
            message = text_data_json["message"]
            input_ids = self.tokenizer(message, return_tensors="pt").input_ids

            outputs = self.model.generate(input_ids)
            response = self.tokenizer.decode(outputs[0]).replace('<pad>', '').replace('</s>', '')


            await self.send(text_data=json.dumps([
                {
                    "type": False,
                    "value": response
                }
            ]))
        