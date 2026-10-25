Load in Google Embedding Projector: https://projector.tensorflow.org
1. Click Load (left panel).
2. Step 1: choose <folder>\vectors.tsv. Step 2: choose <folder>\metadata.tsv. Close the dialog.
3. 'Color by': act, kind, synthetic_picked_by or real_picked_by. 'Label by': label.
4. Choose T-SNE (bottom left) for clusters, or PCA for global distances. Search a label, e.g. 's27'.

Folders:
  all            all 889 points; colour by synthetic_picked_by / real_picked_by to compare the engines
  rag_synthetic  chunks RAG retrieved on the 18 synthetic questions + the questions
  jev_synthetic  chunks JEV + OKF retrieved on the 18 synthetic questions + the questions
  rag_real       chunks RAG retrieved on the 13 real-document questions + the questions
  jev_real       chunks JEV + OKF retrieved on the 13 real-document questions + the questions
Vectors: multilingual-e5-small, 384 dimensions. Rebuild: python scripts\export_projector_tsv.py
