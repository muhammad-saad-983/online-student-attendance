from pydantic import BaseModel


class Student(BaseModel):
    roll_no: str
    person_id: str
    name: str
