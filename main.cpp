/*
 * This code trains a simple GAN (without convolutional layers) on the MNIST dataset.
 * The PyTorch C++ docs (https://pytorch.org/cppdocs/frontend.html) were used to write this code
 *  therefore the code does contain a lot of similarities with the provided examples.
 */

#include <iostream>
#include <torch/torch.h>

// Define the Generator and Discriminator architectures
struct GeneratorNet : torch::nn::Module {
    GeneratorNet(int latent_size, int h_size) {
        this->latent_size = latent_size;
        this->h_size = h_size;
        fc1 = register_module("fc1", torch::nn::Linear(latent_size, h_size));
        fc2 = register_module("fc2", torch::nn::Linear(h_size, h_size));
        fc3 = register_module("fc3", torch::nn::Linear(h_size, 784));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Forward pass through the layers of the model
        x = torch::selu(fc1->forward(x));
        x = torch::selu(fc2->forward(x));
        x = torch::sigmoid(fc3->forward(x));

        // Reshape to a batch of images
        x = x.reshape({x.size(0), 1, 28, 28});
        return x;
    }

    /**
     * Generates a batch of images using the model
     * @param batchSize Desired size of the batch of images
     * @param cuda Whether to generate CUDA tensors
     * @return
     */
    torch::Tensor generateBatch(int batchSize, bool cuda) {
        //Generate a batch of latent vectors from the standard normal distribution
        torch::Tensor z = torch::normal(0, 1, {batchSize, latent_size});
        if (cuda) {
            z = z.cuda();
        }

        // Generate images using the GeneratorNet
        return this->forward(z);
    }


    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    int h_size, latent_size;
};

struct DiscriminatorNet : torch::nn::Module {
    DiscriminatorNet(int h_size) {
        this->h_size = h_size;
        fc1 = register_module("fc1", torch::nn::Linear(784, h_size));
        fc2 = register_module("fc2", torch::nn::Linear(h_size, h_size));
        fc3 = register_module("fc3", torch::nn::Linear(h_size, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Reshape from batch of images to batch of vectors
        x = x.reshape({x.size(0), 784});

        // Forward pass through the layers of the model
        x = torch::selu(fc1->forward(x));
        x = torch::selu(fc2->forward(x));
        // No Sigmoid here since we're outputting logits for better numerical stability
        x = fc3->forward(x);

        return x;
    }


    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    int h_size;
};


int main(int argc, char* argv[]) {
    bool cuda = false;

    // Check for cuda flag
    if (argc > 1 and strcmp("--cuda", argv[1]) == 0) {
        cuda = true;
    }

    auto device = torch::Device(torch::kCPU);

    if (cuda) {
        device = torch::Device(torch::kCUDA);
    }

    auto generator = std::make_shared<GeneratorNet>(128, 256);
    auto discriminator = std::make_shared<DiscriminatorNet>(256);

    if (cuda) {
        generator->to(device);
        discriminator->to(device);
    }

    // Initialize the dataloader (as seen in the example code)
    auto data_loader = torch::data::make_data_loader(
            torch::data::datasets::MNIST("../data/MNIST/raw", torch::data::datasets::MNIST::Mode::kTrain).map(
                    torch::data::transforms::Stack<>()),
            /*batch_size=*/64);

    // Initialize both optimizers
    torch::optim::AdamW optimizer_g(generator->parameters(), 0.0001);
    torch::optim::AdamW optimizer_d(discriminator->parameters(),0.0001);

    // Init variables
    torch::Tensor fake_batch, real_batch, pred_fake, pred_real, loss;

    int step = 0;

    for (int epoch = 1; epoch <= 10; ++epoch) {

        // Acquire real batches from the dataloader
        for (auto& batch : *data_loader) {

            real_batch = batch.data.to(device);

            // ============= Train D ===================
            optimizer_d.zero_grad();

            fake_batch = generator->generateBatch(64, cuda);
            pred_fake = discriminator->forward(fake_batch);
            pred_real = discriminator->forward(real_batch);

            // Compute loss
            loss = torch::binary_cross_entropy_with_logits(pred_fake, torch::zeros_like(pred_fake));
            loss += torch::binary_cross_entropy_with_logits(pred_real, torch::ones_like(pred_real));

            float d_loss = loss.item<float>();

            // Back-propagate the error to get the gradients for the weights
            loss.backward();

            // Update the discriminator
            optimizer_d.step();

            // ============= Train G ===================
            optimizer_g.zero_grad();

            fake_batch = generator->generateBatch(64, cuda);
            pred_fake = discriminator->forward(fake_batch);

            // Compute loss
            loss = torch::binary_cross_entropy_with_logits(pred_fake, torch::ones_like(pred_fake));
            float g_loss = loss.item<float>();

            // Back-propagate the error to get the gradients for the weights
            loss.backward();

            // Update the discriminator
            optimizer_g.step();

            if (step % 1000 == 0) {
                std::cout << "Step " << step <<": losses \ng_loss: " << g_loss << "\nd_loss: " << d_loss << std::endl;
                torch::save(fake_batch.data(), "../samples.pt");
            }

            step++;


        }
    }

    return 0;
}
