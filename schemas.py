from pydantic import BaseModel


class CharacterResponse(BaseModel):
    speech: str
    motion: str
