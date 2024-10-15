from marshmallow import Schema, fields

class UserSchema(Schema):
    id = fields.Int(attribute='id')
    name = fields.Str(attribute='name')
    email = fields.Str(attribute='email')
