#include "particle.h"

// ParticleFilter

ParticleFilter::ParticleFilter(hn_t hn, Hmm *hmm, std::size_t N):
    hn(hn), hmm(hmm), N(N),
    generator(std::chrono::system_clock::now().time_since_epoch().count()) {}

void ParticleFilter::initialise(double y0)
{
    n = 0;
    for (std::size_t i = 0; i < N; i++) {
        double x0 = sample_q0();
        Particle particle;
        particle.w = hmm->p0(x0) * hmm->gn(x0, y0, 0) / q0(x0);
        particle.x.push_back(x0);
        particles.push_back(particle);
    }
}

void ParticleFilter::step(double y_next)
{
    n++;

    double pmf_sum = 0;
    std::vector<double> cmf(N);
    for (std::size_t i = 0; i < N; i++) {
        pmf_sum += particles[i].w;
        cmf[i] = pmf_sum;
    }
    double Wn = pmf_sum;

    J_sampler = std::uniform_real_distribution<double>(0, Wn);

    std::vector<Particle> new_particles;
    for (std::size_t i = 0; i < N; i++) {
        std::size_t j = 0;
        double u = J_sampler(generator);
        while (cmf[j] < u) j++;

        Particle new_particle(particles[j]);
        new_particle.w = Wn / N;
        double xn = new_particle.x[new_particle.x.size()-1];
        double x_next = sample_qn(xn, n);

        double un = hmm->fn(xn, x_next, n);
        un *= hmm->gn(x_next, y_next, n);
        un /= qn(xn, x_next, n);

        new_particle.x.push_back(x_next);
        new_particle.w *= un;
        new_particles.push_back(new_particle);
    }
    particles = new_particles;
}

double ParticleFilter::estimate()const
{
    double numerator, denominator;
    for (const auto &particle : particles) {
        numerator += particle.w * hn(particle.x, n);
        denominator += particle.w;
    }
    return numerator / denominator;
}


// ParticleFilterGaussian

ParticleFilterGaussian::ParticleFilterGaussian(
        hn_t hn, Hmm *hmm, std::size_t N,
        double a, double b):
    ParticleFilter(hn, hmm, N),
    a(a), b(b) {}

double ParticleFilterGaussian::q0(double x)
{
    return (1/std::sqrt(2*M_PI)*b)*
        std::exp(-0.5*std::pow(x/b, 2));
}

double ParticleFilterGaussian::qn(double x_prev, double x, int n)
{
    // n unused - fn constant with n
    return (1/std::sqrt(2*M_PI)*b)*
        std::exp(-0.5*std::pow((x - a*x_prev)/b, 2));
}

double ParticleFilterGaussian::sample_q0()
{
    distribution = std::normal_distribution<double>(0, b);
    return distribution(generator);
}

double ParticleFilterGaussian::sample_qn(double xn_prev, int n)
{
    distribution = std::normal_distribution<double>(a*xn_prev, b);
    return distribution(generator);
}

// ParticleFilterUniform

ParticleFilterUniform::ParticleFilterUniform(
        hn_t hn, Hmm *hmm, std::size_t N,
        double width):
    ParticleFilter(hn, hmm, N),
    width(width) {}

double ParticleFilterUniform::q0(double x)
{
    if (x > -width/2 && x < width/2) {
        return 1/width;
    } else {
        return 0;
    }
}

double ParticleFilterUniform::qn(double x_prev, double x, int n)
{
    if (x > x_prev - width/2 && x < x_prev + width/2) {
        return 1/width;
    } else {
        return 0;
    }
}

double ParticleFilterUniform::sample_q0()
{
    distribution = std::uniform_real_distribution<double>(-width/2, width/2);
    return distribution(generator);
}

double ParticleFilterUniform::sample_qn(double xn_prev, int n)
{
    distribution = std::uniform_real_distribution<double>(xn_prev - width/2, xn_prev + width/2);
    return distribution(generator);
}
