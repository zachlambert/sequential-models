
#include <iostream>
#include <Eigen/Dense>

int main()
{
    Eigen::MatrixXd S_in(3, 3);
    std::cout << S_in.rows() << "x" << S_in.cols() << std::endl;
    S_in << 3, 0, 0,
            0, 2, 0,
            0, 0, 1;
    Eigen::MatrixXd cov = S_in*S_in.transpose();

    Eigen::EigenSolver<Eigen::MatrixXd> eig(cov);
    std::size_t n = eig.eigenvalues().size();

    Eigen::MatrixXd sqrt_eigenvalues(n, n);
    for (std::size_t i = 0; i < n; i++) {
        if (eig.eigenvalues()(i).imag() != 0) {
            std::cerr << "Invalid covariance matrix" << std::endl;
            return 1;
        }
        sqrt_eigenvalues(i, i) = sqrt(eig.eigenvalues()(i).real());
    }
    Eigen::MatrixXd S = sqrt_eigenvalues * eig.eigenvectors().real();
    std::cout << S << std::endl;
    return 0;
}
