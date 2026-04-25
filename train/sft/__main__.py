"""Allow `python -m train.sft` to invoke the SFT trainer entry point."""
from train.sft.train import main


if __name__ == '__main__':
    main()
