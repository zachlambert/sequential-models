#include "hmm.h"
#include "particle.h"
#include <iostream>
#include <memory>

double hn(const std::vector<double> &x, int n)
{
    return x[x.size()-1];
}

int main()
{
    double a=0.8, b=0.5, c=1, d=0.5;

    HmmSamplerGaussian hmm_sampler(a, b, c, d);

    Hmm *hmm = new HmmGaussian(a, b, c, d);

    // Particle filter doesn't have to use the same
    // parameters a an b as the actual model.
    // However, as the difference between the true
    // and trial pdf increases, the estimate variance
    // increases.
    std::size_t N = 100;
    ParticleFilterUniform filter(hn, hmm, N, 5);

    hmm_sampler.initialise();
    filter.initialise(hmm_sampler.yn);
    for (int i = 0; i < 50; i++) {
        std::cout
            << hmm_sampler.n << " "
            << hmm_sampler.xn << " "
            << hmm_sampler.yn << " "
            << filter.estimate()
            << std::endl;
        hmm_sampler.step();
        filter.step(hmm_sampler.yn);
    }
    return 0;
}
