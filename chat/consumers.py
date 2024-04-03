# chat/consumers.py
import json

from channels.generic.websocket import AsyncWebsocketConsumer
from transformers import T5Tokenizer, T5ForConditionalGeneration


class ChatConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
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