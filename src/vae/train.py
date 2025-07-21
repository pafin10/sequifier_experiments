
def train(autoencoder, train_loader, optimizer, loss_function, num_epochs=10, device='cpu'):
    autoencoder.to(device)
    autoencoder.train()
    losses = []

    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(data.size(0), -1)  # Flatten the input
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch = autoencoder(data)
            loss = loss_function(recon_batch, data)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

    return losses

