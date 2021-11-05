from data_loader import DataLoader
import sys

def main(target:str, dest: str)-> None:
    loader = DataLoader()
    loader.load_csv(target)
    loader.feather_df(dest)
    print("Compressing of", target, "to directory", dest, "is completed.")
    return

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])