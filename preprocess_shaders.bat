pushd %~dp0
cd src\shaders
..\..\ShaderPreprocessor.exe scene.frag
popd

shader_minifier.exe -v -o src\shaders\shaders.inl src\shaders\preprocessed.scene.frag src\shaders\scene.vert src\shaders\fxaa.frag
