function(add_example name)
    add_executable(${name}
        src/${name}.cpp
    )
    target_link_libraries(${name} cpp_ml)
endfunction()

add_example(hmm_sample)
add_example(particle)
