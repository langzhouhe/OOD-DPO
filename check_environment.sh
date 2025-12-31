#!/bin/bash

echo "=========================================="
echo "OOD-DPO Environment Check"
echo "=========================================="
echo ""

# Check Python packages
echo "Checking Python packages..."
python -c "
import sys

# Check numpy
try:
    import numpy as np
    version = np.__version__
    print(f'✓ NumPy {version}', end='')
    if version.startswith('1.'):
        print(' (correct version for RDKit)')
    else:
        print(' [WARNING: Should be 1.x for RDKit compatibility]')
        sys.exit(1)
except ImportError:
    print('✗ NumPy not installed')
    sys.exit(1)

# Check tensorboard
try:
    import tensorboard
    print(f'✓ TensorBoard {tensorboard.__version__}')
except ImportError:
    print('✗ TensorBoard not installed')
    sys.exit(1)

# Check rdkit
try:
    from rdkit import Chem
    print('✓ RDKit installed and working')
except ImportError:
    print('✗ RDKit not installed')
    sys.exit(1)
except AttributeError as e:
    print(f'✗ RDKit incompatible with numpy: {e}')
    sys.exit(1)

# Check unicore
try:
    import unicore
    print('✓ Unicore installed')
except ImportError:
    print('✗ Unicore not installed')
    sys.exit(1)

# Check unimol
try:
    import unimol
    from unimol.models import UniMolModel
    print('✓ Uni-Mol installed')
except ImportError:
    print('✗ Uni-Mol not installed')
    print('  Run: cd /home/ubuntu/OOD-DPO/Uni-Mol/unimol && pip install -e .')
    sys.exit(1)

# Check minimol
try:
    from minimol import Minimol
    print('✓ Minimol installed')
except ImportError:
    print('✗ Minimol not installed [WARNING: May not be needed]')

print('')
print('✓✓✓ All critical dependencies OK ✓✓✓')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "✗ Environment check FAILED"
    echo "=========================================="
    echo ""
    echo "Please fix the issues above before running experiments."
    echo "See ENVIRONMENT_FIX.md for detailed instructions."
    exit 1
fi

echo ""

# Check for data files
echo "Checking data files..."
if [ -f "./data/raw/lbap_general_ec50_scaffold.json" ]; then
    echo "✓ Example dataset found: lbap_general_ec50_scaffold.json"
else
    echo "⚠ Example dataset not found (will be needed for experiments)"
fi

# Check for Uni-Mol weights
if [ -f "./weights/mol_pre_no_h_220816.pt" ]; then
    echo "✓ Uni-Mol pretrained weights found"
else
    echo "⚠ Uni-Mol weights not found at: ./weights/mol_pre_no_h_220816.pt"
    echo "  (will be needed for Uni-Mol experiments)"
fi

echo ""

# Check CUDA
echo "Checking CUDA..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠ CUDA not available (CPU mode only)')
"

echo ""
echo "=========================================="
echo "✓✓✓ Environment ready! ✓✓✓"
echo "=========================================="
echo ""
echo "You can now run experiments:"
echo "  - Quick test:  bash test_finetuning.sh"
echo "  - Full exp:    bash run_finetune_comparison.sh"
echo ""
