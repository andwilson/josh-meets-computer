import React from "react";
import Link from "gatsby-link";
import styled from "styled-components";
import Img from "gatsby-image";

//import boy from '../../images/boy.png';
//import computer from '../../images/computer.png';

const Wrapper = styled.div`
  width: 300px;
`;

const TitleBar = styled.div`
  display: flex;
  justify-content: space-between;
  height: 50px;
  margin-bottom: 25px;
`;

const NavBar = styled.div`
  display: flex;
  justify-content: space-between;
  height: 50px;
`;

export default ({ data }) => (
  <Wrapper>
    <TitleBar>
      <Img
        style={{
          width: "60px"
        }}
        sizes={data.boy.sizes}
      />
      <h3>Josh Meets Computer</h3>
      <Img
        style={{
          width: "60px"
        }}
        sizes={data.computer.sizes}
      />
    </TitleBar>
    <NavBar>
      <Link to={"/projects/"}>Projects</Link>
      <Link to={"/notes/"}>Notes</Link>
      <Link to={"/letters/"}>Letters</Link>
      <Link to={"/about/"}>About</Link>
    </NavBar>
  </Wrapper>
);
