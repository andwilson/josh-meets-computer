import React from "react";
import Helmet from "react-helmet";
import styled from "styled-components";

const Title = styled.h1`
  color: black;
  border-bottom: 1px grey solid;
`;

export default () => (
  <div>
    <Helmet title="About" />
    <Title>About</Title>
    <p>Need an enthusiastic engineer on your data science, machine learning or A.I team? Please reach out to me! I am always excited at the prospect of collaborating with other passionately driven people on bigger projects. </p>
  </div>
);
