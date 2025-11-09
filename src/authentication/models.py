from django.db import models


class User(models.Model):
    username = models.CharField(max_length=150, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'authentication_user'

    def __str__(self):
        return self.username

