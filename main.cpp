#include <iostream>
#include <cstdint>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>


double ReLU(double x) {
    return std::max(0.0, x);
}

class NeuralNetwork {
private:
    std::vector<uint32_t> layers;
    std::size_t network_size

    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;

    std::vector<Eigen::VectorXd> value;

    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;

public:
    NeuralNetwork(uint32_t input_node,
                  const std::vector<uint32_t>& hidden_layer,
                  uint32_t output_node)
        : rng(std::random_device{}()), dist(0.0, 1.0)
    {
        // Weight and bias initialisation (randomised)
        // Xavier/He initiallisation in the future

        layers.push_back(input_node);
        layers.insert(layers.end(), hidden_layer.begin(), hidden_layer.end());
        layers.push_back(output_node);

        std::size_t network_size = layers.size();

        for (std::size_t i = 0; i < network_size - 1; ++i) {
            Eigen::MatrixXd W(layers[i], layers[i + 1]);

            for (int r = 0; r < W.rows(); ++r)
                for (int c = 0; c < W.cols(); ++c)
                    W(r, c) = dist(rng);

            weights.push_back(W);
            biases.push_back(Eigen::VectorXd::Zero(layers[i + 1]));
        }
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& inputs) {
        value.clear();

        value.push_back(inputs);

        for (std::size_t i = 1; i < network_size; ++i)
            value.push_back(Eigen::VectorXd::Zero(layers[i]));

        // Calculate value for each node
        for (uint32_t layer = 1; layer < network_size - 1; layer++) {
            for (uint32_t node = 0; node < layers[layer]; node++) {
                const Eigen::VectorXd& x = value[layer - 1];
                auto W = weights[layer - 1].col(node);
                double b = biases[layer - 1](node);

                double z = x.dot(W) + b;
                value[layer](node) = ReLU(z);
            }
        }

        return value.back();
    }

};

int main() {
    constexpr uint32_t INPUT_NODE = 2;
    std::vector<uint32_t> HIDDEN_LAYER = {2};
    constexpr uint32_t OUTPUT_NODE = 2;

    NeuralNetwork net(INPUT_NODE, HIDDEN_LAYER, OUTPUT_NODE);

    Eigen::VectorXd input(2);
    input << 1, 0;

    std::cout << net.forward(input);


    return 0;
}
