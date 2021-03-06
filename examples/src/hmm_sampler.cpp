#include "hmm.h"
#include <iostream>
#include <memory>

int main()
{
    HmmSamplerGaussian hmm_sampler(1, 0.2, 1, 0.1);

    hmm_sampler.initialise();
    for (int i = 0; i < 100; i++) {
        std::cout
            << hmm_sampler.n << " "
            << hmm_sampler.xn << " "
            << hmm_sampler.yn
            << std::endl;
        hmm_sampler.step();
    }
    return 0;
}
