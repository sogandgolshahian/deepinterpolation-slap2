import numpy as np
import matplotlib.pyplot as plt

# === Load data ===
loss = np.load(r"C:\path\to\loss.npy") # Change this
val_loss = np.load(r"C:\path\to\val_loss.npy") # Change this

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(loss, label="Training Loss", linewidth=2)
plt.plot(val_loss, label="Validation Loss", linewidth=2)

plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Training and Validation Loss for Neuron with Calcium Biosensor", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# === Save plot ===
plt.savefig("updated_losses_condition4.png", dpi=300)
plt.show()
