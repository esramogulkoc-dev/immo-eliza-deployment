import joblib
import os
import sys

# Script'in çalıştığı dizin. (Örn: .../immo-eliza-deployment/backend)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model klasörü, şu anda çalıştığınız dizinle aynı seviyede.
# Yani: CURRENT_DIR / models / rf_train_columns.pkl
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
COL_FILE = os.path.join(MODELS_DIR, "rf_train_columns.pkl")


def read_column_names():
    """rf_train_columns.pkl dosyasını yükler ve sütun adlarını ekrana basar."""
    
    # Hata ayıklama çıktısı
    print(f"DEBUG: Sütun dosyası için aranan yol: {COL_FILE}")
    
    if not os.path.exists(COL_FILE):
        print(f"HATA: Sütun dosyası bulunamadı. Beklenen yol: {COL_FILE}")
        return

    try:
        # Sütun listesini yükle
        train_columns = joblib.load(COL_FILE)
        
        print("\n--- rf_train_columns.pkl İÇERİĞİ (SÜTUN ADLARI LİSTESİ) ---")
        # Listeyi daha okunabilir formatta yazdır
        for i, col in enumerate(train_columns):
            print(f"[{i:02d}] {col}")
        print("----------------------------------------------------------\n")
        
        print(f"Toplam {len(train_columns)} sütun adı listelendi. Lütfen bu çıktıyı bana kopyalayıp gönderin.")

    except Exception as e:
        print(f"HATA: Dosya yüklenirken bir sorun oluştu: {e}")
        print(f"Hata detayı: {e}")

if __name__ == "__main__":
    read_column_names()