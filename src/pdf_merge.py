from PyPDF2 import PdfMerger

def pdfs_zusammenfuegen(dateipfade, ausgabe_datei):
    """
    Fügt mehrere PDF-Dateien zusammen.

    :param dateipfade: Liste von Pfaden zu den PDF-Dateien
    :param ausgabe_datei: Pfad zur Ausgabedatei (z. B. 'zusammengefuegt.pdf')
    """
    prefix = "C:/Users/Philipp/Dropbox/6. Semester/Bachelor/literatur/Industrieroboter/"
    dateipfade = [prefix + "Industrieroboter_BEGINN.pdf", prefix + "Industrieroboter_Titel.pdf", prefix + "_Impressum.pdf", prefix + "Industrieroboter_Vorwort.pdf",
                  prefix + "Industrieroboter_Inhaltsverzeichnis.pdf", prefix + "Industrieroboter_1 Einleitung.pdf", prefix + "Industrieroboter_2 Grundlagen der industriellen Robotik.pdf",
                  prefix + "Industrieroboter_3 Technische Machbarkeit.pdf", prefix + "Industrieroboter_4 Wirtschaftlichkeitsbetrachtung.pdf",
                  prefix + "Industrieroboter_5 Konzeption und Planung.pdf", prefix + "Industrieroboter_6 Integration.pdf", prefix + "Industrieroboter_7 Trends.pdf",
                  prefix + "Industrieroboter_Formelzeichen.pdf", prefix + "Industrieroboter_Institutsprofil.pdf", prefix + "Industrieroboter_Stichwortverzeichnis.pdf"]
    ausgabe_datei = "../Industrie-Roboter Planung Integration Trends.pdf"
    merger = PdfMerger()
    
    for pfad in dateipfade:
        try:
            merger.append(pfad)
            print(f"{pfad} hinzugefügt.")
        except Exception as e:
            print(f"Fehler beim Hinzufügen von {pfad}: {e}")

    merger.write(ausgabe_datei)
    merger.close()
    print(f"PDFs erfolgreich zusammengefügt: {ausgabe_datei}")


# Beispielverwendung:
if __name__ == "__main__":
    pdf_liste = ["dokument1.pdf", "dokument2.pdf", "dokument3.pdf"]
    pdfs_zusammenfuegen(pdf_liste, "zusammengefuegt.pdf")
