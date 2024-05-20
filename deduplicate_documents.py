from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    return parser.parse_args()

def main(args):
    with open(args.path) as fh:
        lines = [line.strip() for line in fh.readlines()]
    
    # Deduplicate stripped lines.
    lines = list(set(lines))

    with open(args.path, "w") as fh:
        for line in lines:
            fh.write(line)
            fh.write("\n")
    
    print("Done!")

if __name__ == "__main__":
    main(parse_args())