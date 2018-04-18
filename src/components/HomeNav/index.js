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
  > h1 {
    font-size: 18px;
  }
`;

const NavBar = styled.div`
  display: flex;
  justify-content: space-between;
  height: 50px;
`;

const SLink = styled(Link)`
  font-family: open sans;
  text-decoration: none;
  color: black;
  font-size: 16px;
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
      <h1>Josh Meets Computer</h1>
      <Img
        style={{
          width: "60px"
        }}
        sizes={data.computer.sizes}
      />
    </TitleBar>
    <NavBar>
      <SLink to={"/projects/"}>Projects</SLink>
      <SLink to={"/notes/"}>Notes</SLink>
      <SLink to={"/letters/"}>Letters</SLink>
      <SLink to={"/about/"}>About</SLink>
    </NavBar>
  </Wrapper>
);
