from pathlib import Path

def limpiar_ficheros_uvl(directorio_base):
    # Buscamos recursivamente todos los .uvl
    path_base = Path(directorio_base)
    
    for fichero in path_base.rglob("*.uvl"):
        # Leemos el contenido
        contenido = fichero.read_text(encoding="utf-8")
        
        # Buscamos los delimitadores
        inicio_tag = "```uvl"
        fin_tag = "```"
        
        if inicio_tag in contenido and fin_tag in contenido:
            # Extraemos lo que hay entre las etiquetas
            # split(inicio_tag)[1] coge lo que viene después de ```uvl
            # split(fin_tag)[0] coge lo que viene antes del cierre de ```
            nuevo_contenido = contenido.split(inicio_tag)[1].split(fin_tag)[0]
            
            # Guardamos el archivo limpio (usando .strip() para quitar saltos de línea extra)
            fichero.write_text(nuevo_contenido.strip(), encoding="utf-8")
            print(f"Procesado: {fichero.name}")
        else:
            print(f"Saltado (sin etiquetas): {fichero.name}")

# Uso del script
limpiar_ficheros_uvl("../resources/models/generated/")
