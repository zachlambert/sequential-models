function(add_example name)
    add_executable(${name}
        src/${name}.cpp
    )
    target_link_libraries(${name} cpp_ml)
endfunction()

add_example(hmm_sampler)
add_example(hmm_multivariate_sampler)
add_example(particle_gaussian)
add_example(particle_uniform)

add_example(eigen_test)
