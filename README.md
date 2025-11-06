# Plataforma de autenticación biométrica

## Descripción

Esta es una plataforma de autenticación biométrica que permite a los usuarios autenticarse usando su voz y su imagen.

## Tecnologías utilizadas

- Django
- PostgreSQL
- SpeechBrain
- Torch
- Torchaudio
- Hugging Face

## Instalación

1. Clonar el repositorio
2. Crear y activar un entorno virtual
3. Instalar las dependencias
4. Ejecutar las migraciones
5. Ejecutar el servidor

## Uso

1. Registrar un nuevo usuario
2. Iniciar sesión
3. Autenticar usando su voz y su imagen
4. Verificar la autenticación

## Uso de modelos

Para usar los modelos, se debe ejecutar el siguiente comando:

```bash
# Antes de entrenar los modelos, se debe crear el directorio de datos
make prepare-data

make train-voice
make train-facial
```

Esto entrenará los modelos de voz y facial respectivamente.
