#ifndef CPP_ML_PARTICLE_H
#define CPP_ML_PARTICLE_H

#include "hmm.h"
#include <iostream>
#include <memory>
#include <random>

struct Particle {
    double w;
    std::vector<double> x;
};

class ParticleFilter {
public:
    typedef double (*hn_t)(const std::vector<double> &x, int n);

    ParticleFilter(hn_t hn, Hmm *hmm, std::size_t N):
        hn(hn), hmm(hmm), N(N),
        generator(std::chrono::system_clock::now().time_since_epoch().count())
        {}
    void initialise(double y0)
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

    void step(double y_next)
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

    double estimate()
    {
        double numerator, denominator;
        for (const auto &particle : particles) {
            numerator += particle.w * hn(particle.x, n);
            denominator += particle.w;
        }
        return numerator / denominator;
    }

protected:
    virtual double sample_q0()=0;
    virtual double sample_qn(double x_prev, int n)=0;
    virtual double q0(double x0)=0;
    virtual double qn(double x_prev, double x, int n)=0;
private:
    hn_t hn;
    std::unique_ptr<Hmm> hmm;
    int n; // Current step
    std::size_t N; // Number of particles
    std::vector<Particle> particles;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> J_sampler;
};

class ParticleFilterGaussian: public ParticleFilter {
public:
    ParticleFilterGaussian(
            hn_t hn, Hmm *hmm, std::size_t N,
            double a, double b):
        ParticleFilter(hn, hmm, N),
        a(a), b(b),
        generator(std::chrono::system_clock::now().time_since_epoch().count()) 
    {}

    double q0(double x)
    {
        return (1/std::sqrt(2*M_PI)*b)*
            std::exp(-0.5*std::pow(x/b, 2));
    }
    double qn(double x_prev, double x, int n)
    {
        // n unused - fn constant with n
        return (1/std::sqrt(2*M_PI)*b)*
            std::exp(-0.5*std::pow((x - a*x_prev)/b, 2));
    }
    virtual double sample_q0()
    {
        distribution = std::normal_distribution<double>(0, b);
        return distribution(generator);
    }
    virtual double sample_qn(double xn_prev, int n)
    {
        distribution = std::normal_distribution<double>(a*xn_prev, b);
        return distribution(generator);
    }
private:
    double a, b;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
};

#endif
