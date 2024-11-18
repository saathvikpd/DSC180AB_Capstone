import numpy as np
import torch

class GPTQ:
    def __init__(self, layer, bits=4):
        """
        GPTQ quantizer for Linear layers with error correction.
        """
        self.layer = layer
        self.bits = bits
        self.weight = layer.weight.detach().cpu().numpy()
        self.bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None
        self.Hessian_diag = None  # To store the diagonal of the Hessian
        self.quantized_weights = None
        self.scale = None

    def add_batch(self, input_data):
        """
        Estimate the Hessian diagonal based on layer activations.
        """
        print(f"Estimating Hessian diagonal for layer with shape {self.weight.shape}...")
        # Ensure input_data is numpy.ndarray
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()

        # Compute activations
        activations = input_data @ self.weight.T

        # Compute the Hessian diagonal as the squared sum of activations
        self.Hessian_diag = np.sum(
            (activations[:, :, None] * input_data[:, None, :]) ** 2,
            axis=0
        ).flatten()
        print("Hessian diagonal estimated.")



    # def quantize(self):
    #     """
    #     Perform GPTQ quantization with error correction.
    #     """
    #     print(f"Starting GPTQ quantization for layer with shape {self.weight.shape}...")
    #     max_val = np.max(np.abs(self.weight))
    #     self.scale = max_val / (2 ** (self.bits - 1) - 1)
    
    #     # Initialize quantized weights and error correction term
    #     quantized_weights = np.zeros_like(self.weight)
    #     errors = np.zeros_like(self.weight)

    #     # Compute the Hessian approximation
    #     hessian_matrix = np.diag(self.Hessian_diag)  # Simplified to diagonal if no full Hessian is available
    
    #     # Compute the inverse Hessian using Cholesky decomposition
    #     try:
    #         L = np.linalg.cholesky(hessian_matrix)
    #         inverse_hessian = np.linalg.inv(L.T) @ np.linalg.inv(L)
    #     except np.linalg.LinAlgError:
    #         print("Hessian is not positive definite. Falling back to diagonal inverse.")
    #         inverse_hessian = np.diag(1 / (self.Hessian_diag + 1e-8))
    
    #     # Quantize each weight row-by-row
    #     for i in range(self.weight.shape[0]):
    #         row = self.weight[i, :]
    #         hessian_diag_row = self.Hessian_diag[: len(row)]  # Slice to match row size
            
    #         # Error correction for each element
    #         for j in range(len(row)):
    #             corrected_weight = row[j] + errors[i, j]
    #             quantized_value = np.round(corrected_weight / self.scale)
    #             quantized_value = np.clip(quantized_value, -(2**(self.bits-1)), 2**(self.bits-1) - 1)
    #             quantized_weights[i, j] = quantized_value * self.scale
    
    #             # Update error
    #            # Update error
    #             if j < len(hessian_diag_row):  # Ensure no out-of-bounds access
    #                 # Compute quantization error for the current weight
    #                 quantization_error = (quantized_weights[i, j] - corrected_weight)
                    
    #                 # Redistribute the error to subsequent weights in the same row
    #                 errors[i, j:] += quantization_error / hessian_diag_row[j]

    
    #     self.quantized_weights = quantized_weights
    #     print("GPTQ quantization completed.")


    def quantize(self):
        """
        Perform GPTQ quantization using the Cholesky decomposition for the inverse Hessian.
        """
        print(f"Starting GPTQ quantization for layer with shape {self.weight.shape}...")
        max_val = np.max(np.abs(self.weight))
        self.scale = max_val / (2 ** (self.bits - 1) - 1)
    
        # Initialize quantized weights and error correction term
        quantized_weights = np.zeros_like(self.weight)
        errors = np.zeros_like(self.weight)
    
        # Compute the Hessian approximation
        hessian_matrix = np.diag(self.Hessian_diag)  # Simplified to diagonal if no full Hessian is available
    
        # Compute the inverse Hessian using Cholesky decomposition
        try:
            L = np.linalg.cholesky(hessian_matrix)
            inverse_hessian = np.linalg.inv(L.T) @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            print("Hessian is not positive definite. Falling back to diagonal inverse.")
            inverse_hessian = np.diag(1 / (self.Hessian_diag + 1e-8))

        inverse_hessian = np.diag(inverse_hessian).reshape(self.weight.shape)
    
        # Quantize each weight row-by-row
        for i in range(self.weight.shape[0]):  # Iterate over rows
            row = self.weight[i, :]
    
            # Iterate through each weight in the row
            for j in range(len(row)):
                # Apply error correction to the weight
                corrected_weight = row[j] + errors[i, j]
                
                # Quantize the corrected weight
                quantized_value = np.round(corrected_weight / self.scale)
                quantized_value = np.clip(quantized_value, -(2**(self.bits-1)), 2**(self.bits-1) - 1)
                quantized_weights[i, j] = quantized_value * self.scale
    
                # Compute the quantization error
                quantization_error = (quantized_weights[i, j] - corrected_weight) / inverse_hessian[i, j]
    
                # Redistribute the error using the inverse Hessian
                if inverse_hessian.shape[0] > j:  # Ensure no out-of-bounds access

                    errors[i, j:] += quantization_error * inverse_hessian[i, j:]
    
        self.quantized_weights = quantized_weights
        print("GPTQ quantization using Cholesky inverse completed.")


    def get_quantized_weights(self):
        """
        Return the quantized weights.
        """
        return self.quantized_weights

    def get_quantized_bias(self):
        """
        Return the quantized bias (if present).
        """
        if self.bias is not None:
            return self.bias  # Optionally quantize bias here
        return None
