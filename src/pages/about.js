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
  </div>
);
