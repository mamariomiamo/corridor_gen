# corridor_gen
To use this library:
1. clone the repository in the project directory include/
2. Add the following to CMakeLists.txt of the project
```
add_subdirectory(include/corridor_gen)
target_link_libraries(${PROJECT_NAME}_XX 
  ...
  corridor_gen
  )
```
