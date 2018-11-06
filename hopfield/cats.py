def main():
    from util import BitmapConverter
    bc = BitmapConverter(100)
    # bc.get_images()
    # bc.convert_all()
    bc.to_hopfield_input()


if __name__ == "__main__":
    main()
