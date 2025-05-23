# Federated Learning with Privacy Protection

## Overview

This experiment's federated averaging algorithm is modified based on the open-source project [PFLlib](https://github.com/TsingZ0/PFLlib)

## Environment

- Python 3.8
- PyTorch 2.1.2

## Privacy Protection Features

The federated learning algorithm incorporates privacy protection mechanisms that can be configured in `main.py`:

1. **Gradient Blinding**  
   When `default=False`, blinding operations are disabled  
   When `default=True`, blinding operations are enabled for privacy protection  

   ```python
   parser.add_argument('-sg', "--enable_signing", type=bool, default=False,
                        help="Enable signing and verification for client parameters")
   ```

2. **Signature Verification**

 When `default=False`, signature and verification operations are disabled
 When `default=True`, signature verification is performed on client parameters

```python
 parser.add_argument('-sg', "--enable_signing", type=bool, default=False,
                     help="Enable signing and verification for client parameters") 
```

