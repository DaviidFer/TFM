"""Validation toolbox bajo app/validation.

Este paquete agrupa los cuatro módulos de validación (monos, correlación,
forward y estabilidad). Los consumidores deben importar directamente desde
cada submódulo (`from app.validation.monos import ...`), igual que hace
`app/services/validation_service.py`.

Mantenemos este paquete sin reexportar nombres para evitar una fachada
ancha que solo añadiría ruido y posibles imports accidentales.
"""
