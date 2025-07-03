from PIL import Image
import pillow_avif  # wichtig: importieren, damit AVIF unterstützt wird

def avif_zu_png(eingabe_datei, ausgabe_datei):
    """
    Wandelt eine AVIF-Datei in eine PNG-Datei um.

    :param eingabe_datei: Pfad zur AVIF-Datei (z. B. 'bild.avif')
    :param ausgabe_datei: Pfad zur Ausgabe-PNG (z. B. 'bild.png')
    """
    try:
        with Image.open(eingabe_datei) as img:
            img.save(ausgabe_datei, format="PNG")
            print(f"Erfolgreich konvertiert: {eingabe_datei} → {ausgabe_datei}")
    except Exception as e:
        print(f"Fehler bei der Konvertierung: {e}")


# Beispielverwendung:
if __name__ == "__main__":
    avif_zu_png("C:/Users/Philipp/Desktop/gitProjects/visualkinematics_v2/latex_v2/images/lbr2.avif", "C:/Users/Philipp/Desktop/gitProjects/visualkinematics_v2/latex_v2/images/lbr2.png")
