import os
import zipfile

def download_mura():
    
    print("Downloading MURA dataset from Kaggle...")
    
    
    os.system('kaggle datasets download -d cjinny/mura-v11 -p ./data')
    
    print("Extracting files...")
    
    zip_path = './data/mura-v11.zip'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./data/')
        
    
    os.remove(zip_path)
    print("Dataset ready in the /data/ folder!")

def download_figshare_dataset():
    print("Downloading Bone Tumor dataset from Figshare...")
    url = "https://figshare.com/ndownloader/files/50653575"
    zip_path = "./data/bone_tumor_dataset.zip"
    
    # Download the file in chunks (good practice for large files)
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./data/bone_tumor')
        
    os.remove(zip_path)
    print("Bone Tumor Dataset ready!")

if __name__ == "__main__":
    # Create a data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    download_mura()
    download_figshare_dataset()